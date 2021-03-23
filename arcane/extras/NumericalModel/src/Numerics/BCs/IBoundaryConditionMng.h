// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef IBOUNDARYCONDITIONMNG_H
#define IBOUNDARYCONDITIONMNG_H

#include <map>

#include <arcane/ItemGroupRangeIterator.h>
#include <arcane/ItemUniqueId.h>

#include "Numerics/BCs/BoundaryConditionTypes.h"

/*!
  \class IBoundaryConditionMng
  \author Daniele A. Di Pietro <daniele-antonio.di-pietro@ifp.fr>
  \date 2007-08-03
  \brief Base class for boundary condition manager service
*/

class IBoundaryCondition;

class IBoundaryConditionMng {
 public:
  //! Boundary condition pointer
  typedef IBoundaryCondition* BoundaryConditionPtr;

  //! Face unique id to boundary condition map
  typedef std::map<ItemUniqueId, BoundaryConditionPtr> 
    FaceUid2BoundaryConditionPtrMap;

 public:
  virtual ~IBoundaryConditionMng() {}

 public:
  //! Initialize
  virtual void init() = 0;

 public:
  //! Return boundary condition with selected tag
  virtual BoundaryConditionPtr boundaryCondition(const String& tag) = 0;
  //! Return boundary condition associated to a face
  virtual BoundaryConditionPtr getBoundaryCondition(const Face& face) = 0;
  //! Return the group of faces with boundary condition of the type passed
  //! as an argument
  virtual FaceGroup getFacesOfType(BoundaryConditionTypes::eType type) = 0;
  //! Return the group of faces with boundary condition of type different
  //! from the one passed as an argument
  virtual FaceGroup getFacesNotOfType(BoundaryConditionTypes::eType type) = 0;
};

#endif
