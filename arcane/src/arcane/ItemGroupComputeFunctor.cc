// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGroupComputeFunctor.cc                                  (C) 2000-2016 */
/*                                                                           */
/* Functors de calcul des éléments d'un groupe en fonction d'un autre groupe */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/Item.h"
#include "arcane/ItemGroupComputeFunctor.h"
#include "arcane/ItemGroupImpl.h"
#include "arcane/ItemGroup.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/Properties.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
//! Calcul des entités propres du groupe
void OwnItemGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  ItemGroup parent(m_group->parent());

  m_group->beginTransaction();
  Int32UniqueArray items_lid;

  ENUMERATE_ITEM(iitem,parent){
    Item item = *iitem;
    if (item.isOwn())
      items_lid.add(iitem.itemLocalId());
  }
  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "OwnItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void GhostItemGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  ItemGroup parent(m_group->parent());
  m_group->beginTransaction();
  Int32UniqueArray items_lid;

  ENUMERATE_ITEM(iitem,parent){
    Item item = *iitem;
    if (!item.isOwn())
      items_lid.add(iitem.itemLocalId());
  }
  
  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "GhostItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void InterfaceItemGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());

  if (family->itemKind() != IK_Face){
    trace->fatal() << "InterfaceGroupComputeFunctor::executeFunctor()"
                   << " Incoherent family expected=" << IK_Face << " current=" << family->itemKind();
  }

  m_group->beginTransaction();
  Int32UniqueArray items_lid;


  ENUMERATE_FACE(iface,parent) {
    Face face = *iface;
    Cell bcell = face.backCell();
    Cell fcell = face.frontCell();
    if ( !bcell.null() && !fcell.null() ) {
      const bool isBackOwn = bcell.isOwn();
      const bool isFrontOwn = fcell.isOwn();
      if ( isBackOwn != isFrontOwn ) {
        items_lid.add(face.localId());
      }
    }
  }

  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "InterfaceItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Fonctor pour calculer un groupe contenant les entités connectées
 * aux entités du groupe parent.
 */
template<typename ItemType> void
ItemItemGroupComputeFunctor<ItemType>::
executeFunctor()
{
  IMesh* mesh = m_group->mesh();
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());

  eItemKind ik = ItemTraitsT<ItemType>::kind();
  if (ik!=family->itemKind()){
    ARCANE_FATAL("Incoherent family wanted={0} current={1} v={2}",
                 ik,family->itemKind(),m_group->itemKind());
  }

  m_group->beginTransaction();
  Int32UniqueArray items_lid;
  Int32UniqueArray markers(family->maxLocalId());
  markers.fill(0);
  ItemType* null_type = 0;
  ENUMERATE_ITEM(iitem,parent){
    ItemInternal* i = (*iitem).internal();
    for( ItemInternalEnumerator iitem2(i->internalItems(null_type)); iitem2.hasNext(); ++iitem2 ){
      Int32 lid = iitem2.localId();
      if (markers[lid]==0){
        markers[lid] = 1;
        items_lid.add(lid);
      }
    }
  }
  bool do_sort = mesh->properties()->getBoolWithDefault("sort-subitemitem-group",false);

  m_group->setItems(items_lid,do_sort);
  m_group->endTransaction();

  trace->debug() << "ItemItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size()
                 << " mysize2=" << items_lid.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void InnerFaceItemGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());
  IItemFamily* parent_family = parent.itemFamily();
  m_group->beginTransaction();
  Int32UniqueArray items_lid;

  if (parent.isAllItems()) {
    ENUMERATE_FACE(iface,family->allItems()){
      const Face& face = *iface;
      if (!face.isSubDomainBoundary())
        items_lid.add(face.localId());
    }
  }
  else {
    BoolUniqueArray markers(parent_family->maxLocalId());
    markers.fill(false);
    ENUMERATE_CELL(icell,parent) {
      markers[icell.localId()] = true;
    }
    ENUMERATE_FACE(iface,family->allItems()) {
      Face face = *iface;
      Cell bcell = face.backCell();
      Cell fcell = face.frontCell();
      if ((!bcell.null() && markers[bcell.localId()]) &&
          (!fcell.null() && markers[fcell.localId()]))
        items_lid.add(face.localId());
    }
  }

  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "InnerFaceItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OuterFaceItemGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());
  IItemFamily* parent_family = parent.itemFamily();
  m_group->beginTransaction();
  Int32UniqueArray items_lid;

  if (parent.isAllItems()) {
    ENUMERATE_FACE(iface,family->allItems()){
      const Face& face = *iface;
      if (face.isSubDomainBoundary())
        items_lid.add(face.localId());
    }
  }
  else {
    BoolUniqueArray markers(parent_family->maxLocalId());
    markers.fill(false);
    ENUMERATE_CELL(icell,parent) {
      markers[icell.localId()] = true;
    }
    ENUMERATE_FACE(iface,family->allItems()) {
      Face face = *iface;
      Cell bcell = face.backCell();
      Cell fcell = face.frontCell();
      if ((!bcell.null() && markers[bcell.localId()]) ^
          (!fcell.null() && markers[fcell.localId()]))
        items_lid.add(face.localId());
    }
  }

  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "OuterFaceItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ActiveCellGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());

  if (family->itemKind() != IK_Cell){
    trace->fatal() << "AllActiveCellGroupComputeFunctor::executeFunctor()"
                   << " Incoherent family expected=" << IK_Cell << " current=" << family->itemKind();
  }

  m_group->beginTransaction();
  Int32UniqueArray items_lid;
  ENUMERATE_CELL(iitem,parent){
    Cell item = *iitem;
    if(item.isActive())
      items_lid.add(item.localId());
  }
  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "AllActiveCellGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OwnActiveCellGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());

  if (family->itemKind() != IK_Cell){
    trace->fatal() << "OwnActiveCellGroupComputeFunctor::executeFunctor()"
                   << " Incoherent family expected=" << IK_Cell << " current=" << family->itemKind();
  }

  m_group->beginTransaction();

  Int32UniqueArray items_lid;
  ENUMERATE_CELL(iitem,parent){
    Cell item = *iitem;
    if(item.isOwn() && item.isActive())
      items_lid.add(item.localId());
  }
  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "OwnActiveCellGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LevelCellGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());

  if (family->itemKind() != IK_Cell){
    trace->fatal() << "LevelCellGroupComputeFunctor::executeFunctor()"
                   << " Incoherent family expected=" << IK_Cell << " current=" << family->itemKind();
  }

  m_group->beginTransaction();
  Int32UniqueArray items_lid;

  ENUMERATE_CELL(iitem,parent){
    Cell item = *iitem;
    if(item.level() == m_level)
      items_lid.add(item.localId());
  }
  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "LevelCellGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OwnLevelCellGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());

  m_group->beginTransaction();
  Int32UniqueArray items_lid;
  if (family->itemKind() != IK_Cell){
    trace->fatal() << "LevelCellGroupComputeFunctor::executeFunctor()"
                   << " Incoherent family expected=" << IK_Cell << " current=" << family->itemKind();
  }
  ENUMERATE_CELL(iitem,parent){
    Cell item = *iitem;
    if (item.isOwn() && (item.level() == m_level))
      items_lid.add(item.localId());
  }
  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "OwnLevelItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ActiveFaceItemGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());
  IItemFamily* parent_family = parent.itemFamily();
  m_group->beginTransaction();
  Int32UniqueArray items_lid;
  Integer counter_0=0;
  Integer counter_1=0;
  Integer counter_2=0;
  Integer counter = 0;
  if (parent.isAllItems()) {
    ENUMERATE_FACE(iface,family->allItems()){
      Face face = *iface;
      counter++;
      if (face.isSubDomainBoundary()) {
        counter_0++;
        if (face.boundaryCell().isActive()){
          items_lid.add(face.localId());
          counter_1++;
        }
      }
      else if (face.backCell().isActive() && face.frontCell().isActive()){
        items_lid.add(face.localId());
        counter_2++;
      }
    }
  }
  else {
    BoolUniqueArray markers(parent_family->maxLocalId());
    markers.fill(false);
    ENUMERATE_CELL(icell,parent) {
      markers[icell.localId()] = true;
    }
    ENUMERATE_FACE(iface,family->allItems()) {
      const Face & face = *iface;
      if (face.isSubDomainBoundary()) {
        if (face.boundaryCell().isActive() && markers[face.boundaryCell().localId()])
          items_lid.add(face.localId());
      }
      else if ( (face.backCell().isActive() && face.frontCell().isActive())  &&
                (markers[face.backCell().localId()] || markers[face.frontCell().localId()]))
        items_lid.add(face.localId());
    }
  }
  trace->debug() << "NUMBER OF ALL FACES= " << counter
    		         <<  "\n NUNMBER OF BOUNDARY FACES= "  << counter_0
    		         << "\n NUMBER OF ACTIVE BOUNDARY FACES= "  << counter_1
    		         << "\n NUMBER OF ACTIVE INTERIOR FACES= " << counter_2 << "\n";
  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "ActiveFaceItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OwnActiveFaceItemGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());
  IItemFamily* parent_family = parent.itemFamily();
  m_group->beginTransaction();
  Int32UniqueArray items_lid;

  if (parent.isAllItems()) {
    ENUMERATE_FACE(iface,family->allItems()){
      const Face& face = *iface;
      if (face.isSubDomainBoundary()) {
        if (face.isOwn() && face.boundaryCell().isActive())
          items_lid.add(face.localId());
      }
      else if (face.isOwn() && face.backCell().isActive() && face.frontCell().isActive()){
        items_lid.add(face.localId());
      }
    }
  } else {
    BoolUniqueArray markers(parent_family->maxLocalId());
    markers.fill(false);
    ENUMERATE_CELL(icell,parent) {
      markers[icell.localId()] = true;
    }
    ENUMERATE_FACE(iface,family->allItems()) {
      const Face & face = *iface;
      if (face.isSubDomainBoundary()) {
        if (face.isOwn() && face.boundaryCell().isActive() && markers[face.boundaryCell().localId()])
          items_lid.add(face.localId());
      }
      else if (face.isOwn() &&
               markers[face.backCell().localId()] && face.backCell().isActive() &&
               markers[face.frontCell().localId()] && face.frontCell().isActive())
        items_lid.add(face.localId());
    }
  }

  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "OwnActiveFaceItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void InnerActiveFaceItemGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());
  IItemFamily* parent_family = parent.itemFamily();
  m_group->beginTransaction();
  Int32UniqueArray items_lid;

  if (parent.isAllItems()) {
    ENUMERATE_FACE(iface,family->allItems()){
      Face face = *iface;
      if (face.isSubDomainBoundary())
        continue;

      if (face.backCell().isActive() && face.frontCell().isActive())
        items_lid.add(face.localId());
    }
  }
  else {
    BoolUniqueArray markers(parent_family->maxLocalId());
    markers.fill(false);
    ENUMERATE_CELL(icell,parent) {
      markers[icell.localId()] = true;
    }
    ENUMERATE_FACE(iface,family->allItems()) {
      Face face = *iface;
      if (face.isSubDomainBoundary())
        continue;

      Cell bcell = face.backCell();
      Cell fcell = face.frontCell();
      if (((markers[bcell.localId()]) && markers[fcell.localId()]) &&
          (bcell.isActive() && fcell.isActive()))
        items_lid.add(face.localId());
    }
  }

  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "InnerActiveFaceItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OuterActiveFaceItemGroupComputeFunctor::
executeFunctor()
{
  ITraceMng* trace = m_group->mesh()->traceMng();
  IItemFamily* family = m_group->itemFamily();
  ItemGroup parent(m_group->parent());
  IItemFamily* parent_family = parent.itemFamily();
  m_group->beginTransaction();
  Int32UniqueArray items_lid;

  if (parent.isAllItems()) {
    ENUMERATE_FACE(iface,family->allItems()){
      const Face& face = *iface;
      if (face.isSubDomainBoundary() && face.boundaryCell().isActive())
        items_lid.add(face.localId());
    }
  }
  else {
    BoolUniqueArray markers(parent_family->maxLocalId());
    markers.fill(false);
    ENUMERATE_CELL(icell,parent) {
      markers[icell.localId()] = true;
    }
    ENUMERATE_FACE(iface,family->allItems()) {
      Face face = *iface;
      if (face.isSubDomainBoundary()){
        Cell bcell= face.boundaryCell();
        if(bcell.isActive() && markers[bcell.localId()])
          items_lid.add(face.localId());
      }
    }
  }

  m_group->setItems(items_lid);
  m_group->endTransaction();

  trace->debug() << "OuterActiveFaceItemGroupComputeFunctor::execute()"
                 << " this=" << m_group
                 << " parent_name=" << parent.name()
                 << " name=" << m_group->name()
                 << " parent_count=" << parent.size()
                 << " mysize=" << m_group->size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template class ItemItemGroupComputeFunctor<Node>;
template class ItemItemGroupComputeFunctor<Edge>;
template class ItemItemGroupComputeFunctor<Face>;
template class ItemItemGroupComputeFunctor<Cell>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
