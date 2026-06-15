/*
 * Minimal soc1sim U-mode crt for the hart report probe.
 *
 * Unlike the shared example crt, this keeps the firmware-provided a0 launch
 * argument live and passes it directly to main() on every hart.
 */

#if __has_include("erbium/isa/syscall.h")
#include "erbium/isa/syscall.h"
#else
#include "erbium-soc1sim/isa/syscall.h"
#endif

  .section .text.init, "ax", @progbits
  .global _boot
_boot:
  la    sp, __stack_base
  csrr  t0, hartid
  andi  t0, t0, 0x3F
  lui   t1, %hi(STACK_SIZE)
  addi  t1, t1, %lo(STACK_SIZE)
  mul   t0, t0, t1
  sub   sp, sp, t0
  j _start

  .text
  .global _start
_start:
  .option push
  .option norelax
1:auipc gp, %pcrel_hi(__global_pointer$)
  addi  gp, gp, %pcrel_lo(1b)
  .option pop

  csrr  t0, hartid
  andi  t0, t0, 0x3F
  lui   t1, %hi(NUM_HARTS)
  addi  t1, t1, %lo(NUM_HARTS)
  bgeu  t0, t1, .Lskip_main

  call  main
  j     .Lkernel_exit

.Lskip_main:
  mv    a0, zero

.Lkernel_exit:
  mv    a1, a0
  li    a2, KERNEL_RETURN_SUCCESS
  li    a0, SYSCALL_RETURN_FROM_KERNEL
  ecall

1:j 1b
