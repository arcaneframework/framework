// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef NUMERICALDOMAINIMPL_H_
#define NUMERICALDOMAINIMPL_H_

#include "NumericalModel/SubDomainModel/INumericalDomain.h"

#include <arcane/utils/FatalErrorException.h>
#include <arcane/utils/ITraceMng.h>

#include <map>

using namespace Arcane;

class INumericalModel;

// A l'heure actuelle, 
// m_cell_group est le groupe de toutes les mailles du domaine
// m_face_group est le groupe de toutes les faces du domaine => toutes les faces de m_cell_group
// m_internal_face_group est le groupe de toutes les faces internes du domaine => m_face_group sans les faces de bords
// m_boundary_face_group est un groupe de faces de bord du domaine => CL sur les bords utilisateurs

// A REVOIR CAR NON COHERENT
// Pour être correct, on donne m_cell_group. Les groupes m_face_group et m_internal_face_group s'en déduisent.
// ( CE QUI SUIT N'EST PAS FAIT DANS LES SCHEMAS DE COORES (de la faute de Sylvain) )
// Dès lors, m_boundary_face_group devrait être m_face_group.outerFaces() 
// c'est à dire m_internal_face_group + m_boundary_face_group = m_face_group

// Le problème est que l'on donne en plus les conditions limites utilisateurs dans ce domaine. 
// Il y a les intersections de groupes à gérer 

// EN FAIT, CETTE CLASSE SE RETROUVE PARTOUT ET N'A PAS DE DOC :(

class NumericalDomainImpl : public INumericalDomain
{
public :
     

  typedef std::pair<FaceGroup,Integer> FaceBoundaryType ;
  typedef std::map<Integer,FaceGroup>::iterator FaceBoundaryIter ;
  typedef std::map<Integer,FaceGroup>::const_iterator FaceBoundaryConstIter ;
  
  NumericalDomainImpl(ITraceMng * trace_mng,
                      INumericalModel * parent = NULL) 
    : m_trace_mng(trace_mng)
    , m_parent(parent)
    , m_name("DefaultDomain")
    , m_has_boundary(false)
    , m_has_interface(false)
  {
    ARCANE_ASSERT((m_trace_mng),("Trace manager pointer null"));
  }
  
  virtual ~NumericalDomainImpl() {}
 
  INumericalGraph * getNumericalGraph() const { throw FatalErrorException("NumericalDomainImpl don't use numerical graph"); }
  
  INumericalModel * getParent() const { return m_parent; }

  const String& getName() const { return m_name; }
  
  void setName(String name) { m_name = name; }
    
  void setInternalCells(CellGroup cell_group) { m_cell_group = cell_group; }
  void setAllFaces(FaceGroup face_group) { m_face_group = face_group; }
  
  void setInternalItems(CellGroup cell_group, FaceGroup internal_face)
  {
    m_cell_group = cell_group;
    m_internal_face_group = internal_face;
  }
  
  bool hasBoundaryItems() { return m_has_boundary; }
  
  void setBoundary(FaceGroup boundary_face)
  {
    m_has_boundary = true;
    m_boundary_face_group = boundary_face;
  }

  void setCellBoundary(CellGroup cell_group)
  {
    m_boundary_cell_group = cell_group;
  }

  //record on boundary with an id key
  void addFaceBoundary(FaceGroup boundary_face, Integer id) 
  { 
    std::map<Integer,FaceGroup>::iterator it = m_boundaries.find(id);
    
    if(it != m_boundaries.end()) 
    {
      m_trace_mng->fatal() << "boundary id already used";
    }
    
    m_boundaries[id] = boundary_face;
  }
  
  FaceBoundaryIter getFaceBoundaryIter() {
      return m_boundaries.begin() ;
  }
  

  FaceBoundaryConstIter getFaceBoundaryIter() const {
      return m_boundaries.begin() ;
  }
  
  bool hasInterfaceItems() { return m_has_interface; }
  
  void setInterfaceItems(CellGroup overlap_group, 
                         FaceGroup overlap_internal_face,
                         CellGroup external_group)
  {
    m_has_interface = true;
    m_overlap_group = overlap_group;
    m_overlap_internal_face_group = overlap_internal_face;
    m_external_group = external_group;
  }
  
  CellGroup internalCells() const { return m_cell_group; }
  CellGroup externalCells() const { return m_overlap_group; }
  FaceGroup allFaces() const { return m_face_group; }
  FaceGroup internalFaces() const { return m_internal_face_group; }
  FaceGroup boundaryFaces() const { return m_boundary_face_group; }
  CellGroup boundaryCells() const { return m_boundary_cell_group; }
  CellGroup interfaceCells() const { return m_external_group; }
  FaceGroup internalInterfaceFaces() const { return m_overlap_internal_face_group; }
  
  // Potentiellement, il est possible d'utiliser un conteneur m_boundaries sans index donné par l'utilisateur
  // La gestion peut se faire en interne
  // A voir et modifier le cas échéant
  // Utiliser de préférence un énumérateur
  
  //JMG : non il ne faut pas changer index en id : id est une clé pas un index
  Integer nbBoundary() const { return m_boundaries.size(); }
  
  // Utiliser de préférence un énumérateur
  // JMG : non car id est un clé
  FaceGroup boundary(Integer id) const
  {
    std::map<Integer,FaceGroup>::const_iterator it = m_boundaries.find(id);
    
    if(it == m_boundaries.end()) return m_null_boundary;
    
    return it->second;
  }

  void printInfo() const;
  
private :
  
  ITraceMng * m_trace_mng;

  INumericalModel * m_parent;
  
  String m_name;

  CellGroup m_cell_group;
  FaceGroup m_face_group;
  FaceGroup m_internal_face_group;
  
  bool m_has_boundary;
  FaceGroup m_boundary_face_group;
  CellGroup m_boundary_cell_group;
  
  std::map<Integer,FaceGroup> m_boundaries;
  const FaceGroup m_null_boundary;

  bool m_has_interface;
  CellGroup m_overlap_group;
  FaceGroup m_overlap_internal_face_group;
  CellGroup m_external_group;
};

#endif /*NUMERICALDOMAINIMPL_H_*/
