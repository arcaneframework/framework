#ifndef ISURFACE_H
#define ISURFACE_H

#include <arcane/utils/ArcaneGlobal.h>

ARCANE_BEGIN_NAMESPACE
NUMERICS_BEGIN_NAMESPACE

//! Purely virtual interface for surface representation
/*! Use as a pretty 'void*' pointer. Each implementation has to cast this 
 *  object before using it
 */
class ISurface
{
public:
  virtual ~ISurface() {}
};

NUMERICS_END_NAMESPACE
ARCANE_END_NAMESPACE

#endif /* ISURFACE_H */
