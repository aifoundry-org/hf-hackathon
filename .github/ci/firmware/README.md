# Sys-emu firmware blobs

Minimal `esperanto-fw` ELF set required by `erbium_soc1sim_argbuf_dynmem --device sys_emu`:

- `BootromTrampolineToBL2/BootromTrampolineToBL2.elf`
- `ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf`
- `MachineMinion/MachineMinion.elf`
- `MasterMinion/MasterMinion.elf`
- `WorkerMinion/WorkerMinion.elf`

`.github/ci/scripts/install_firmware.sh` copies these into `${ET_INSTALL}/lib/esperanto-fw/`.
Override with `ET_FIRMWARE_DIR` or `ET_FIRMWARE_URL` if you do not use the vendored tree.

These firmware blobs are owned by us and distributed under Apache-2.0 with this
repository.
