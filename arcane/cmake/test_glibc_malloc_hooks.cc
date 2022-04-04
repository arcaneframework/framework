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
