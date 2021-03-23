// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef ILINEARSYSTEMVISITOR_H
#define ILINEARSYSTEMVISITOR_H

#include <arcane/ItemTypes.h>
#include <arcane/utils/FatalErrorException.h>
#include <arcane/utils/TraceInfo.h>

using namespace Arcane;

/**
 * Interface du service du modu?le de schma numeric.
 */

class ILinearSystem;
class HypreLinearSystem;

class ILinearSystemVisitor
{
  
public:
  virtual ~ILinearSystemVisitor() { }

  /** 
   *  Initialise 
   */
  virtual void init() = 0;
  
  virtual bool visit(    ILinearSystem * system) { throw FatalErrorException(A_FUNCINFO,"not implemented");  }
  virtual bool visit(HypreLinearSystem * system) { throw FatalErrorException(A_FUNCINFO,"not implemented");  }
};
 //END_NAME_SPACE_PROJECT


#endif
