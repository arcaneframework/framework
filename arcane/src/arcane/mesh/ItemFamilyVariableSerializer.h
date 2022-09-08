// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyVariableSerializer.h                              (C) 2000-2016 */
/*                                                                           */
/* Gère la sérialisation/désérialisation des variables d'une famille.        */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MESH_ITEMFAMILYVARIABLESERIALIZER_H
#define ARCANE_MESH_ITEMFAMILYVARIABLESERIALIZER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"

#include "arcane/IItemFamilySerializeStep.h"

#include "arcane/mesh/MeshGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
class IVariable;
class ItemFamilySerializeArgs;
ARCANE_MESH_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Gère la sérialisation/désérialisation des variables d'une famille.
 */
class ARCANE_MESH_EXPORT ItemFamilyVariableSerializer
: public TraceAccessor
, public IItemFamilySerializeStep
{
 public:
  ItemFamilyVariableSerializer(IItemFamily* family);
  ~ItemFamilyVariableSerializer();
 public:
  void initialize() override;
  void notifyAction(const NotifyActionArgs&) override {}
  void serialize(const ItemFamilySerializeArgs& args) override;
  void finalize() override{}
  ePhase phase() const override { return IItemFamilySerializeStep::PH_Variable; }
  IItemFamily* family() const override { return m_item_family; }
 protected:
  IItemFamily* _family() const { return m_item_family; }
 private:
  IItemFamily* m_item_family;
  /*!
   * \brief Liste des variables à échanger.
    
   IMPORTANT: Cette liste doit être identique pour tous les sous-domaines
   sinon les désérialisations vont donner des résultats incorrects.
  */
  UniqueArray<IVariable*> m_variables_to_exchange;

 private:
  void _serializePartialVariable(IVariable* var,ISerializer* sbuf,Int32ConstArrayView local_ids);
  void _checkSerialization(ISerializer* sbuf,Int32ConstArrayView local_ids);
  void _checkSerializationVariable(ISerializer* sbuf,IVariable* var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_MESH_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
