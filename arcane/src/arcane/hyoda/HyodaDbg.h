/*

  executeFunctor

movl     %rdi,%rdx
testb    $1,%cl
je       0x42bf57

direct_with_offset:
movl     24(%rdi),%rax
movl     16(%rdi),%rdi
movl     (%rax,%rdi,1),%rax
addl     24(%rdx),%rdi
movl     -1(%rcx,%rax,1),%rcx
movl     %rcx,%r11
jmp      *%r11

0x42bf57:
indirect_via_vtable:
movl     16(%rdi),%rdi
addl     24(%rdx),%rdi
movl     %rcx,%r11
jmp      *%r11

jecxz    toTheBeginning

*/
