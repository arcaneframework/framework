// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVectorView.cc                                  (C) 2000-2024 */
/*                                                                           */
/* Vue sur un vecteur sur des entités composants.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/ComponentItemVectorView.h"

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/core/materials/IMeshMaterial.h"
#include "arcane/core/materials/IMeshEnvironment.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshMaterial* MatItemVectorView::
material() const
{
  return static_cast<IMeshMaterial*>(component());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshEnvironment* EnvItemVectorView::
environment() const
{
  return static_cast<IMeshEnvironment*>(component());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemVectorView ComponentItemVectorView::
_subView(Integer begin,Integer size)
{
  Integer nb_total = nbItem();

  // Pas d'éléments, retourne un tableau vide
  if (nb_total==0){
    return ComponentItemVectorView(m_component);
  }

  if (begin>=nb_total){
    // Indice de début supérieur au nombre d'éléments.
    ARCANE_THROW(ArgumentException,"Bad 'begin' value '{0}' total={1}",begin,nb_total);
  }

  ConstituentItemLocalIdListView mn = m_constituent_list_view._subView(begin, size);
  ConstArrayView<MatVarIndex> mvs = _matvarIndexes().subView(begin,size);
  ConstArrayView<Int32> ids = _internalLocalIds().subView(begin,size);

  return { m_component, mvs, mn, ids };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatItemVectorView MatItemVectorView::
_subView(Integer begin,Integer size)
{
  return { component(), ComponentItemVectorView::_subView(begin, size) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvItemVectorView EnvItemVectorView::
_subView(Integer begin,Integer size)
{
  return { component(), ComponentItemVectorView::_subView(begin, size) };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
