﻿// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
/*---------------------------------------------------------------------------*/
/* Ref.cc                                                      (C) 2000-2019 */
/*                                                                           */
/* Gestion des références sur une instance.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Ref.h"

#include <iostream>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arccore
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool RefBase::DeleterBase::
_destroyHandle(void* instance,Internal::ExternalRef& handle)
{
  ARCCORE_UNUSED(instance);
  //std::cerr << "DELETE SERVICE i=" << instance << " h=" << handle << "\n";
  if (handle.isValid())
    return true;
  return false;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arccore

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

