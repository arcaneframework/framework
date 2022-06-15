// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentPartItemVectorView.cc                              (C) 2000-2022 */
/*                                                                           */
/* Vue sur un vecteur sur une partie des entités composants.                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentPartItemVectorView.h"
#include "arcane/core/materials/IMeshMaterial.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatPartItemVectorView::
MatPartItemVectorView(IMeshMaterial* material,const ComponentPartItemVectorView& view)
: ComponentPartItemVectorView(view)
, m_material(material)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvPartItemVectorView::
EnvPartItemVectorView(IMeshEnvironment* env,const ComponentPartItemVectorView& view)
: ComponentPartItemVectorView(view)
, m_environment(env)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
