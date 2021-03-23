// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemVectorView.cc                                  (C) 2000-2015 */
/*                                                                           */
/* Vue sur un vecteur sur des entités composants.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/ArgumentException.h"

#include "arcane/materials/ComponentItemVectorView.h"
#include "arcane/materials/IMeshMaterial.h"
#include "arcane/materials/IMeshEnvironment.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

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
    throw ArgumentException(A_FUNCINFO,
                            String::format("Bad 'begin' value '{0}' total={1}",begin,nb_total));
  }

  ConstArrayView<ComponentItemInternal*> mn = m_items_internal_main_view.subView(begin,size);
  ConstArrayView<MatVarIndex> mvs = matvarIndexes().subView(begin,size);

  return ComponentItemVectorView(m_component,mvs,mn);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatItemVectorView MatItemVectorView::
_subView(Integer begin,Integer size)
{
  return MatItemVectorView(component(),ComponentItemVectorView::_subView(begin,size));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

EnvItemVectorView EnvItemVectorView::
_subView(Integer begin,Integer size)
{
  return EnvItemVectorView(component(),ComponentItemVectorView::_subView(begin,size));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
