// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef BOUNDARYCONDITIONMNGSERVICE_H
#define BOUNDARYCONDITIONMNGSERVICE_H

#include <vector>
#include <map>

#include "Utils/ItemComparator.h"

#include "BoundaryConditionMng_axl.h"

using namespace Arcane;

/*!
  \author Daniele Di Pietro <daniele-antonio.di-pietro@ifp.fr>
  \date 2007-30-7
  \brief Boundary condition manager
*/

class IBoundaryCondition;

class BoundaryConditionMngService : public ArcaneBoundaryConditionMngObject {
 public:
  //! Boundary condition array
  typedef std::vector<BoundaryConditionPtr>                    BoundaryConditionPtrArray;
  //! Tag to boundary condition map
  typedef std::map<String, BoundaryConditionPtr>               Tag2BoundaryConditionPtrMap;  
  //! Face unique id to boundary condition map
  typedef std::map<Face, BoundaryConditionPtr, ItemComparator> Face2BoundaryConditionPtrMap;

 public:
  //! Constructor
  BoundaryConditionMngService(const ServiceBuildInfo& sbi) :
    ArcaneBoundaryConditionMngObject(sbi),
    m_initialized(false) {}

  //! Destructor
  virtual ~BoundaryConditionMngService() {
    for(BoundaryConditionPtrArray::iterator i = m_bcs.begin(); i != m_bcs.end(); i++)
      delete *i;
  }

  //! Return version info
  virtual VersionInfo versionInfo() const { return VersionInfo(1, 0, 0); }

 public:
  //! Initialize
  void init();
  //! Return boundary condition with selected tag
  BoundaryConditionPtr boundaryCondition(const String& tag) { return m_tag_2_bc[ tag ]; }
  //! Given an iterator to a boundary face, return a pointer to the
  //! corresponding boundary condition
  BoundaryConditionPtr getBoundaryCondition(const Face& F);
  //! Return the group of faces with boundary condition of the type passed
  //! as an argument
  FaceGroup getFacesOfType(BoundaryConditionTypes::eType type);
  //! Return the group of faces with boundary condition of type different
  //! from the one passed as an argument
  FaceGroup getFacesNotOfType(BoundaryConditionTypes::eType type);
 private:
  //! Initialized tag
  bool                            m_initialized;

  //! Boundary conditions
  BoundaryConditionPtrArray       m_bcs;
  //! Tag to boundary condition
  Tag2BoundaryConditionPtrMap     m_tag_2_bc; 
  //! Face unique id to boundary condition
  Face2BoundaryConditionPtrMap    m_face_2_bc;
};

#endif
