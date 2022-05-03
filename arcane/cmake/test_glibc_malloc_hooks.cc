// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-

#include <malloc.h>
#include <iostream>

int
main()
{
  std::cout << __malloc_hook << "\n";
  std::cout << __free_hook << "\n";
  std::cout << __realloc_hook << "\n";
  return 0;
}
