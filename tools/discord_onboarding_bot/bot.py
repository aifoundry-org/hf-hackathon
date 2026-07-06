#!/usr/bin/env python3
import asyncio
import logging
import os
import re
from dataclasses import dataclass

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOG = logging.getLogger("core_et_onboarding")
INTRO_PROMPT_MARKER = "Welcome to the CORE-ET hackathon."
LEGACY_INTRO_NUDGE_MARKER = "If you’re here for the CORE-ET hackathon, read the pinned message"
USER_MENTION_RE = re.compile(r"<@!?(\d+)>")


def env_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    token: str
    guild_id: int | None
    welcome_channel_id: int
    welcome_channel_name: str
    intro_channel_id: int
    intro_channel_name: str
    general_channel_id: int | None
    general_channel_name: str
    hackathon_role_id: int | None
    hackathon_role_name: str
    join_emoji: str
    role_add_emoji: str
    role_remove_emoji: str
    target_claims_channel_id: int | None
    target_claims_channel_name: str
    board_help_channel_id: int | None
    board_help_channel_name: str
    participant_form_url: str
    hf_org_url: str
    github_repo_url: str
    discord_event_url: str
    intro_nudge_ttl_seconds: int
    auto_reply_to_intros: bool


def load_config() -> Config:
    token = os.getenv("DISCORD_TOKEN", "").strip()
    welcome = env_int("WELCOME_CHANNEL_ID")
    welcome_name = os.getenv("WELCOME_CHANNEL_NAME", "").strip()
    intro = env_int("INTRO_CHANNEL_ID")
    intro_name = os.getenv("INTRO_CHANNEL_NAME", "").strip()
    missing = []
    if not token:
        missing.append("DISCORD_TOKEN")
    if not intro and not intro_name:
        missing.append("INTRO_CHANNEL_ID or INTRO_CHANNEL_NAME")
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    return Config(
        token=token,
        guild_id=env_int("DISCORD_GUILD_ID"),
        welcome_channel_id=welcome,
        welcome_channel_name=welcome_name,
        intro_channel_id=intro,
        intro_channel_name=intro_name,
        general_channel_id=env_int("GENERAL_CHANNEL_ID"),
        general_channel_name=os.getenv("GENERAL_CHANNEL_NAME", "").strip(),
        hackathon_role_id=env_int("HACKATHON_ROLE_ID"),
        hackathon_role_name=os.getenv("HACKATHON_ROLE_NAME", "Hackathon").strip() or "Hackathon",
        join_emoji=os.getenv("JOIN_EMOJI", "✅").strip() or "✅",
        role_add_emoji=os.getenv("ROLE_ADD_EMOJI", "✅").strip() or "✅",
        role_remove_emoji=os.getenv("ROLE_REMOVE_EMOJI", "❌").strip() or "❌",
        target_claims_channel_id=env_int("TARGET_CLAIMS_CHANNEL_ID"),
        target_claims_channel_name=os.getenv("TARGET_CLAIMS_CHANNEL_NAME", "").strip(),
        board_help_channel_id=env_int("BOARD_HELP_CHANNEL_ID"),
        board_help_channel_name=os.getenv("BOARD_HELP_CHANNEL_NAME", "").strip(),
        participant_form_url=os.getenv("PARTICIPANT_FORM_URL", "").strip(),
        hf_org_url=os.getenv("HF_ORG_URL", "https://huggingface.co/AIFoundry-hackathon").strip(),
        github_repo_url=os.getenv("GITHUB_REPO_URL", "https://github.com/aifoundry-org/hf-hackathon").strip(),
        discord_event_url=os.getenv("DISCORD_EVENT_URL", "").strip(),
        intro_nudge_ttl_seconds=env_int("INTRO_NUDGE_TTL_SECONDS") or 86400,
        auto_reply_to_intros=env_bool("AUTO_REPLY_TO_INTROS", True),
    )


CFG = load_config()

intents = discord.Intents.default()
intents.members = True
intents.guilds = True
intents.guild_messages = True
intents.reactions = True

bot = commands.Bot(command_prefix="!", intents=intents)
WELCOME_MESSAGES: dict[int, int] = {}
INTRO_CONTROL_MESSAGE_IDS: set[int] = set()
INTRO_PROMPT_MESSAGES: dict[int, int] = {}


def find_text_channel(guild: discord.Guild, channel_id: int | None, channel_name: str = "") -> discord.TextChannel | None:
    if channel_id:
        channel = guild.get_channel(channel_id)
        if isinstance(channel, discord.TextChannel):
            return channel
    if channel_name:
        normalized = channel_name.lstrip("#")
        channel = discord.utils.get(guild.text_channels, name=normalized)
        if isinstance(channel, discord.TextChannel):
            return channel
    return None


def channel_url(guild_id: int, channel_id: int | None) -> str | None:
    if channel_id is None:
        return None
    return f"https://discord.com/channels/{guild_id}/{channel_id}"


def channel_mention(channel_id: int | None, channel_name: str = "", fallback: str = "") -> str:
    if channel_id:
        return f"<#{channel_id}>"
    if channel_name:
        return f"#{channel_name.lstrip('#')}"
    return fallback


def resolved_channel_mention(
    guild: discord.Guild,
    channel_id: int | None,
    channel_name: str = "",
    fallback: str = "",
) -> str:
    channel = find_text_channel(guild, channel_id, channel_name)
    if channel is not None:
        return channel.mention
    return channel_mention(channel_id, channel_name, fallback)


def intro_template() -> str:
    return (
        "**GitHub handle:**\n"
        "**Hugging Face handle:**\n"
        "**Track interest:** performance / model port / recipe / demo / just exploring\n"
        "**Looking for teammates:** yes/no\n"
        "**First action you plan to take:**\n"
        "**Other info:**"
    )


def claim_template() -> str:
    return (
        "Handle/team:\n"
        "Track: Llama 3.2 performance / pre-approved model / model port / recipe / demo\n"
        "Target model or benchmark:\n"
        "HF model link, if relevant:\n"
        "Expected proof: sys-emu log / board CI / PR / recipe / demo\n"
        "Need help from organizers:"
    )


def intro_control_text() -> str:
    return (
        "👋 Welcome to the CORE-ET hackathon.\n\n"
        "There are a lot of new people joining in, so help us make this a more welcoming "
        "and easier-to-navigate place by introducing yourself here 🚀\n\n"
        f"React with {CFG.role_add_emoji} to get the `{CFG.hackathon_role_name}` role, then introduce yourself using:\n\n"
        f"{intro_template()}\n\n"
        f"React with {CFG.role_remove_emoji} to remove the `{CFG.hackathon_role_name}` role."
    )


def starter_text(member: discord.abc.User | None = None) -> str:
    hello = f"Hi {member.mention}.\n\n" if member else ""
    target_claims = channel_mention(
        CFG.target_claims_channel_id, CFG.target_claims_channel_name, "#target-claims"
    )
    board_help = channel_mention(CFG.board_help_channel_id, CFG.board_help_channel_name, "#board-help")
    event_line = f"\nLive event / office hours: {CFG.discord_event_url}\n" if CFG.discord_event_url else ""
    form_line = f"\nParticipant form: {CFG.participant_form_url}\n" if CFG.participant_form_url else ""
    return (
        f"{hello}Welcome to the CORE-ET hackathon.\n\n"
        "To participate:\n"
        f"1. Post your intro in {channel_mention(CFG.intro_channel_id, CFG.intro_channel_name, '#hackathon-intros')}.\n"
        f"2. Claim a target in {target_claims}.\n"
        "3. Run sys-emu or board CI.\n"
        "4. Submit a PR, recipe, or demo with proof.\n\n"
        "Good first tracks:\n"
        "- Llama 3.2 1B performance\n"
        "- Pre-approved model performance\n"
        "- New model port\n"
        "- Reproducible recipe / agent workflow\n"
        "- Board-backed demo\n\n"
        f"General discussion: {channel_mention(CFG.general_channel_id, CFG.general_channel_name, '#hf-hackathon')}\n"
        f"Board help: {board_help}\n"
        f"HF org: {CFG.hf_org_url}\n"
        f"GitHub repo: {CFG.github_repo_url}"
        f"{form_line}"
        f"{event_line}"
    )


def welcome_embed(member: discord.Member) -> discord.Embed:
    embed = discord.Embed(
        title="Welcome to the CORE-ET hackathon",
        description=(
            f"If you are here for the hackathon, react with {CFG.join_emoji} so we "
            "can count you in and send the intro template."
        ),
        color=0x3B82F6,
    )
    embed.add_field(
        name="What happens next",
        value=(
            f"You will get the `{CFG.hackathon_role_name}` role and the intro "
            f"template will be posted in {channel_mention(CFG.intro_channel_id, CFG.intro_channel_name, '#hackathon-intros')}."
        ),
        inline=False,
    )
    embed.add_field(
        name="Useful channels",
        value=(
            f"General: {channel_mention(CFG.general_channel_id, CFG.general_channel_name, '#hf-hackathon')}\n"
            f"Targets: {channel_mention(CFG.target_claims_channel_id, CFG.target_claims_channel_name, '#target-claims')}\n"
            f"Board help: {channel_mention(CFG.board_help_channel_id, CFG.board_help_channel_name, '#board-help')}"
        ),
        inline=False,
    )
    embed.set_footer(text="Click Join hackathon if you are participating.")
    return embed


def welcome_text(member: discord.Member) -> str:
    return (
        f"Hey {member.mention}!.\n\n"
        "If you’re here for the CORE-ET hackathon, head to "
        f"{resolved_channel_mention(member.guild, CFG.intro_channel_id, CFG.intro_channel_name, '#hackathon-intros')}, "
        f"read the pinned message, and react with {CFG.role_add_emoji} to join."
    )


def intro_nudge_text(member: discord.Member) -> str:
    return (
        f"Hey {member.mention}!\n\n"
        f"{intro_control_text()}"
    )


def is_intro_nudge_message(message: discord.Message) -> bool:
    return bool(
        bot.user
        and message.author.id == bot.user.id
        and not message.pinned
        and message.id not in INTRO_CONTROL_MESSAGE_IDS
        and (
            INTRO_PROMPT_MARKER in message.content
            or LEGACY_INTRO_NUDGE_MARKER in message.content
        )
    )


def tagged_user_id(message: discord.Message) -> int | None:
    match = USER_MENTION_RE.search(message.content)
    return int(match.group(1)) if match else None


async def get_or_create_hackathon_role(guild: discord.Guild) -> discord.Role:
    if CFG.hackathon_role_id:
        role = guild.get_role(CFG.hackathon_role_id)
        if role is not None:
            return role
        raise RuntimeError(f"Configured role ID {CFG.hackathon_role_id} was not found")

    role = discord.utils.get(guild.roles, name=CFG.hackathon_role_name)
    if role is not None:
        return role

    return await guild.create_role(
        name=CFG.hackathon_role_name,
        mentionable=False,
        reason="CORE-ET hackathon participant role",
    )


async def post_intro_nudge(member: discord.Member) -> None:
    channel = find_text_channel(member.guild, CFG.intro_channel_id, CFG.intro_channel_name)
    if channel is None:
        LOG.warning("Intro channel %s/%s not found", CFG.intro_channel_id, CFG.intro_channel_name)
        return

    await delete_intro_prompt_for_user(member.id, channel)
    try:
        prompt = await channel.send(
            content=intro_nudge_text(member),
            allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False),
        )
    except discord.Forbidden:
        LOG.warning("Missing permission to post intro prompt for %s in %s", member, channel)
        return
    except discord.HTTPException:
        LOG.warning("Could not post intro prompt for %s in %s", member, channel)
        return
    await ensure_role_reactions(prompt)
    LOG.info("Posted intro prompt for %s (%s) in %s", member, member.id, channel)
    INTRO_PROMPT_MESSAGES[member.id] = prompt.id
    asyncio.create_task(
        delete_intro_nudge_after(
            member.guild.id,
            channel.id,
            prompt.id,
            member.id,
            CFG.intro_nudge_ttl_seconds,
        )
    )


async def delete_intro_nudge_after(
    guild_id: int,
    channel_id: int,
    message_id: int,
    user_id: int,
    delay_seconds: int,
) -> None:
    await asyncio.sleep(delay_seconds)
    guild = bot.get_guild(guild_id)
    if guild is None:
        return
    channel = guild.get_channel(channel_id)
    if not isinstance(channel, discord.TextChannel):
        return
    try:
        message = await channel.fetch_message(message_id)
    except discord.HTTPException:
        return
    if is_intro_nudge_message(message):
        try:
            await message.delete()
            if INTRO_PROMPT_MESSAGES.get(user_id) == message_id:
                INTRO_PROMPT_MESSAGES.pop(user_id, None)
        except discord.HTTPException:
            LOG.warning("Could not delete expired intro nudge message %s", message_id)


async def cleanup_expired_intro_nudges(guild: discord.Guild) -> None:
    channel = find_text_channel(guild, CFG.intro_channel_id, CFG.intro_channel_name)
    if channel is None:
        return

    now_ts = discord.utils.utcnow().timestamp()
    cutoff = now_ts - CFG.intro_nudge_ttl_seconds
    deleted = 0
    scheduled = 0
    try:
        async for message in channel.history(limit=200):
            if not is_intro_nudge_message(message):
                continue
            created_ts = message.created_at.timestamp()
            if created_ts > cutoff:
                asyncio.create_task(
                    delete_intro_nudge_after(
                        guild.id,
                        channel.id,
                        message.id,
                        0,
                        max(1, int(created_ts + CFG.intro_nudge_ttl_seconds - now_ts)),
                    )
                )
                scheduled += 1
                continue
            try:
                await message.delete()
                deleted += 1
            except discord.HTTPException:
                LOG.warning("Could not delete expired intro nudge message %s", message.id)
    except discord.Forbidden:
        LOG.warning("Missing permission to inspect expired intro nudges in %s", channel)
    except discord.HTTPException:
        LOG.warning("Could not inspect expired intro nudges in %s", channel)

    if deleted:
        LOG.info("Deleted %d expired intro nudge messages in %s", deleted, channel)
    if scheduled:
        LOG.info("Scheduled %d existing intro nudge messages for expiry in %s", scheduled, channel)


async def ensure_existing_intro_prompts(guild: discord.Guild) -> None:
    channel = find_text_channel(guild, CFG.intro_channel_id, CFG.intro_channel_name)
    if channel is None:
        return

    repaired = 0
    try:
        async for message in channel.history(limit=100):
            if not is_intro_nudge_message(message):
                continue
            user_id = tagged_user_id(message)
            if user_id is None:
                continue
            member = guild.get_member(user_id)
            if member is None:
                try:
                    member = await guild.fetch_member(user_id)
                except discord.HTTPException:
                    continue
            desired_content = intro_nudge_text(member)
            if message.content != desired_content:
                try:
                    await message.edit(
                        content=desired_content,
                        allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False),
                    )
                    repaired += 1
                except discord.HTTPException:
                    LOG.warning("Could not repair intro prompt message %s", message.id)
                    continue
            await ensure_role_reactions(message)
            INTRO_PROMPT_MESSAGES[user_id] = message.id
    except discord.Forbidden:
        LOG.warning("Missing permission to repair existing intro prompts in %s", channel)
    except discord.HTTPException:
        LOG.warning("Could not inspect intro prompts for repair in %s", channel)

    if repaired:
        LOG.info("Repaired %d existing intro prompt messages in %s", repaired, channel)


async def delete_intro_prompt_for_user(user_id: int, channel: discord.TextChannel) -> None:
    prompt_id = INTRO_PROMPT_MESSAGES.pop(user_id, None)
    candidate_ids = [prompt_id] if prompt_id else []

    for candidate_id in candidate_ids:
        try:
            message = await channel.fetch_message(candidate_id)
        except discord.HTTPException:
            continue
        if not message.pinned:
            try:
                await message.delete()
            except discord.HTTPException:
                LOG.warning("Could not delete intro prompt message %s", candidate_id)

    try:
        async for message in channel.history(limit=25):
            if message.pinned or message.id in INTRO_CONTROL_MESSAGE_IDS:
                continue
            if bot.user and message.author.id != bot.user.id:
                continue
            if f"<@{user_id}>" not in message.content and f"<@!{user_id}>" not in message.content:
                continue
            if (
                INTRO_PROMPT_MARKER not in message.content
                and LEGACY_INTRO_NUDGE_MARKER not in message.content
            ):
                continue
            try:
                await message.delete()
            except discord.HTTPException:
                LOG.warning("Could not delete intro prompt message %s", message.id)
            break
    except discord.Forbidden:
        LOG.warning("Missing permission to inspect intro prompt cleanup history in %s", channel)
    except discord.HTTPException:
        LOG.warning("Could not inspect intro prompt cleanup history in %s", channel)


async def count_in_member(member: discord.Member) -> None:
    role = await get_or_create_hackathon_role(member.guild)
    if role not in member.roles:
        await member.add_roles(role, reason="Reacted to CORE-ET hackathon welcome message")


async def add_hackathon_role(member: discord.Member) -> bool:
    role = await get_or_create_hackathon_role(member.guild)
    if role not in member.roles:
        await member.add_roles(role, reason="Reacted to CORE-ET hackathon role add emoji")
        return True
    return False


async def remove_hackathon_role(member: discord.Member) -> None:
    role = await get_or_create_hackathon_role(member.guild)
    if role in member.roles:
        await member.remove_roles(role, reason="Reacted to CORE-ET hackathon role remove emoji")


async def clear_user_reaction(
    guild: discord.Guild,
    channel_id: int,
    message_id: int,
    emoji: str,
    member: discord.Member,
) -> None:
    channel = guild.get_channel(channel_id)
    if not isinstance(channel, discord.TextChannel):
        return
    try:
        message = await channel.fetch_message(message_id)
        await message.remove_reaction(emoji, member)
    except discord.Forbidden:
        LOG.warning("Missing permission to clear %s reaction on intro role message %s", emoji, message_id)
    except discord.HTTPException:
        LOG.warning("Could not clear %s reaction on intro role message %s", emoji, message_id)


async def apply_role_reaction(member: discord.Member, emoji: str) -> bool:
    if emoji == CFG.role_add_emoji:
        await add_hackathon_role(member)
        return True
    if emoji == CFG.role_remove_emoji:
        await remove_hackathon_role(member)
        return True
    return False


async def ensure_role_reactions(message: discord.Message) -> None:
    desired_reactions = [CFG.role_add_emoji, CFG.role_remove_emoji]
    current_reactions = [str(reaction.emoji) for reaction in message.reactions]

    if current_reactions != desired_reactions:
        try:
            await message.clear_reactions()
        except discord.Forbidden:
            LOG.warning("Missing permission to reorder intro role reactions on message %s", message.id)
        except discord.HTTPException:
            LOG.warning("Could not reorder intro role reactions on message %s", message.id)

    for emoji in desired_reactions:
        try:
            await message.add_reaction(emoji)
        except discord.HTTPException:
            LOG.warning("Could not add %s reaction to intro role message %s", emoji, message.id)


async def ensure_intro_control_message(guild: discord.Guild) -> None:
    channel = find_text_channel(guild, CFG.intro_channel_id, CFG.intro_channel_name)
    if channel is None:
        LOG.warning("Intro channel %s/%s not found", CFG.intro_channel_id, CFG.intro_channel_name)
        return

    markers = ("CORE-ET Hackathon Intro Template", "Welcome to the CORE-ET hackathon.")
    message = None
    try:
        for pinned in await channel.pins():
            if (
                pinned.author == bot.user
                and any(marker in pinned.content for marker in markers)
                and tagged_user_id(pinned) is None
            ):
                message = pinned
                break
        if message is None:
            async for candidate in channel.history(limit=50):
                if (
                    candidate.author == bot.user
                    and any(marker in candidate.content for marker in markers)
                    and tagged_user_id(candidate) is None
                ):
                    message = candidate
                    break
    except discord.Forbidden:
        LOG.warning("Missing permission to inspect intro channel history/pins")

    if message is None:
        message = await channel.send(intro_control_text())
    elif message.content != intro_control_text():
        try:
            await message.edit(content=intro_control_text())
        except discord.Forbidden:
            LOG.warning("Missing permission to edit intro control message %s", message.id)

    INTRO_CONTROL_MESSAGE_IDS.add(message.id)

    try:
        if not message.pinned:
            await message.pin(reason="CORE-ET hackathon intro template")
    except discord.Forbidden:
        LOG.warning("Missing permission to pin intro control message %s", message.id)
    except discord.HTTPException:
        LOG.warning("Could not pin intro control message %s", message.id)

    await ensure_role_reactions(message)


@bot.event
async def on_ready() -> None:
    LOG.info("Logged in as %s (%s)", bot.user, bot.user.id if bot.user else "?")
    if CFG.guild_id:
        guild = discord.Object(id=CFG.guild_id)
        bot.tree.copy_global_to(guild=guild)
        synced = await bot.tree.sync(guild=guild)
        LOG.info("Synced %d guild commands", len(synced))
    else:
        synced = await bot.tree.sync()
        LOG.info("Synced %d global commands", len(synced))

    for guild in bot.guilds:
        await ensure_intro_control_message(guild)
        await ensure_existing_intro_prompts(guild)
        await cleanup_expired_intro_nudges(guild)


@bot.event
async def on_member_join(member: discord.Member) -> None:
    LOG.info("Member joined: %s (%s)", member, member.id)
    await post_intro_nudge(member)


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    if bot.user and payload.user_id == bot.user.id:
        return

    guild = bot.get_guild(payload.guild_id) if payload.guild_id else None
    if guild is None:
        return
    member = guild.get_member(payload.user_id)
    if member is None:
        try:
            member = await guild.fetch_member(payload.user_id)
        except discord.HTTPException:
            LOG.warning("Could not fetch member %s", payload.user_id)
            return

    emoji = str(payload.emoji)
    if payload.message_id in INTRO_CONTROL_MESSAGE_IDS:
        try:
            if await apply_role_reaction(member, emoji):
                await clear_user_reaction(guild, payload.channel_id, payload.message_id, emoji, member)
        except discord.Forbidden:
            LOG.warning("Missing permissions to update role for %s", member)
        except Exception:
            LOG.exception("Failed to update role for %s", member)
        return

    if emoji in {CFG.role_add_emoji, CFG.role_remove_emoji}:
        channel = guild.get_channel(payload.channel_id)
        if isinstance(channel, discord.TextChannel):
            try:
                message = await channel.fetch_message(payload.message_id)
            except discord.HTTPException:
                return
            if is_intro_nudge_message(message):
                target_user_id = tagged_user_id(message)
                if target_user_id is not None:
                    INTRO_PROMPT_MESSAGES[target_user_id] = message.id
                if target_user_id is not None and target_user_id != member.id:
                    await clear_user_reaction(guild, payload.channel_id, payload.message_id, emoji, member)
                    return
                try:
                    if await apply_role_reaction(member, emoji):
                        await clear_user_reaction(guild, payload.channel_id, payload.message_id, emoji, member)
                except discord.Forbidden:
                    LOG.warning("Missing permissions to update role for %s", member)
                except Exception:
                    LOG.exception("Failed to update role for %s", member)
                return

    if emoji != CFG.join_emoji:
        return
    expected_user_id = WELCOME_MESSAGES.get(payload.message_id)
    if expected_user_id is None:
        return
    if payload.user_id != expected_user_id:
        return

    try:
        await count_in_member(member)
        channel = guild.get_channel(payload.channel_id)
        if isinstance(channel, discord.TextChannel):
            try:
                message = await channel.fetch_message(payload.message_id)
                await message.delete()
                WELCOME_MESSAGES.pop(payload.message_id, None)
            except discord.HTTPException:
                LOG.warning("Could not delete welcome message %s", payload.message_id)
    except discord.Forbidden:
        LOG.warning("Missing permissions to assign role to %s", member)
    except Exception:
        LOG.exception("Failed to count in member %s", member)


@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot:
        return
    intro_match = message.channel.id == CFG.intro_channel_id
    if CFG.intro_channel_name:
        intro_match = intro_match or message.channel.name == CFG.intro_channel_name.lstrip("#")
    if intro_match and isinstance(message.channel, discord.TextChannel):
        await delete_intro_prompt_for_user(message.author.id, message.channel)
    if CFG.auto_reply_to_intros and intro_match:
        try:
            await message.add_reaction("👋")
        except discord.HTTPException:
            pass
        await message.reply(
            "Thanks for the intro. Next step: claim one concrete target in "
            f"{channel_mention(CFG.target_claims_channel_id, CFG.target_claims_channel_name, '#target-claims')}.",
            mention_author=True,
            allowed_mentions=discord.AllowedMentions(users=True, roles=False, everyone=False),
        )
    await bot.process_commands(message)


@bot.tree.command(name="start", description="Get the CORE-ET hackathon starter kit.")
@app_commands.describe(dm="Send the starter kit by DM if possible.")
async def start(interaction: discord.Interaction, dm: bool = False) -> None:
    if dm:
        try:
            await interaction.user.send(starter_text(interaction.user))
            await interaction.response.send_message("Sent you the starter kit in DM.", ephemeral=True)
            return
        except discord.Forbidden:
            pass
    await interaction.response.send_message(starter_text(interaction.user), ephemeral=True)


@bot.tree.command(name="intro_template", description="Show the hackathon intro template.")
async def intro_template_command(interaction: discord.Interaction) -> None:
    await interaction.response.send_message(intro_template(), ephemeral=True)


@bot.tree.command(name="claim_template", description="Show the target-claim template.")
async def claim_template_command(interaction: discord.Interaction) -> None:
    await interaction.response.send_message(f"```text\n{claim_template()}\n```", ephemeral=True)


if __name__ == "__main__":
    bot.run(CFG.token)
