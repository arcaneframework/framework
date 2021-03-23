#ifndef TYPESDTLOCAL_H
#define TYPESDTLOCAL_H
#include <arcane/ItemGroup.h>

struct TypesDtLocal
{
  //! type de calcul 
  typedef enum
  {
    GlobalFineDt,
    GlobalCoarseDt,
    DFVF1,
    DFVF2,
    VFVF1,
    VFVF2
  } eComputationType ;
};
#endif /*TYPESTEST_ */
