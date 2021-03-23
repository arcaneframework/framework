#include "arcane/utils/ArcaneGlobal.h"
#include "arcane/utils/Exception.h"

#include <iostream>

extern "C" ARCANE_IMPORT
int func0();

extern "C" ARCANE_EXPORT void
arcaneRegisterAcceleratorRuntimecuda();

int
main(int argc,char* argv[])
{
  int r = 0;
  try{
    arcaneRegisterAcceleratorRuntimecuda();
    ARCANE_UNUSED(argc);
    ARCANE_UNUSED(argv);
    r = func0();
  }
  catch(const Arcane::Exception& e){
    std::cerr << "Exception e=" << e << "\n";
  }
  catch(const std::exception& e){
    std::cerr << "Exception e=" << e.what() << "\n";
  }
  return r;
}
