#include "arcane/utils/ArcaneGlobal.h"

extern "C"
int arcaneTestHip1();
extern "C"
int arcaneTestHip2();
extern "C"
int arcaneTestHip3();
extern "C"
int arcaneTestHipNumArray();
extern "C"
int arcaneTestHipReduction();

extern "C" ARCANE_EXPORT
int func0()
{
  arcaneTestHip1();
  arcaneTestHip2();
  arcaneTestHip3();
  arcaneTestHipNumArray();
  arcaneTestHipReduction();
  return 0;
}
