# CORE-ET Discord Onboarding Bot

Small Discord bot for the CORE-ET hackathon server.

It does these things:

1. Tags each new member in the intro channel with a personal copy of the intro/template message.
2. Maintains a pinned intro/template message with the same join/remove reactions.
3. Assigns or removes the `Hackathon` role when the user clicks the matching reaction.
4. Deletes the per-user intro prompt after the user posts in the intro channel or after 24 hours.
5. Provides `/start`, `/intro_template`, and `/claim_template` slash commands.

It does not auto-DM everyone on join.

## Discord Setup

Create a Discord application and bot in the Developer Portal.

Enable this privileged intent:

- Server Members Intent

The bot does not need Message Content Intent.

Invite the bot with:

- `bot`
- `applications.commands`

Suggested bot permissions:

- View Channels
- Send Messages
- Embed Links
- Manage Roles
- Manage Messages
- Pin Messages
- Add Reactions
- Read Message History

## Environment

Copy `.env.example` to `.env` and fill in:

- `DISCORD_TOKEN`
- `INTRO_CHANNEL_ID` or `INTRO_CHANNEL_NAME`
- `HACKATHON_ROLE_NAME=Hackathon`
- `JOIN_EMOJI=✅`
- `ROLE_ADD_EMOJI=✅`
- `ROLE_REMOVE_EMOJI=❌`

Optional:

- `HACKATHON_ROLE_ID`
- `WELCOME_CHANNEL_ID` or `WELCOME_CHANNEL_NAME`
- `GENERAL_CHANNEL_ID` or `GENERAL_CHANNEL_NAME`
- `TARGET_CLAIMS_CHANNEL_ID` or `TARGET_CLAIMS_CHANNEL_NAME`
- `BOARD_HELP_CHANNEL_ID` or `BOARD_HELP_CHANNEL_NAME`
- `PARTICIPANT_FORM_URL`
- `DISCORD_GUILD_ID`
- `INTRO_NUDGE_TTL_SECONDS`

If you use `HACKATHON_ROLE_ID`, create the role first. If you omit it, the bot
will look for a role named `Hackathon` and create it if missing. The bot role
must be above `Hackathon` in the Discord role list.

## Run

```bash
cd tools/discord_onboarding_bot
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python bot.py
```

## Channel Setup

Recommended channels:

- `#hackathon-intros`
- `#hf-hackathon` or `#hackathon-general`
- `#target-claims`
- `#board-help`
