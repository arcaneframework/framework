// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef INUMERICALDOMAIN_H_
#define INUMERICALDOMAIN_H_

/**
  * \author Jean-Marc GRATIEN
  * \version 1.0
  * \brief Interface des NumericalDomains, manager des objets supports géométriques
  * modélisant la discrétisation du domaine continu sur lequel un modèle
  * numérique s'applique.
  * Ce manageur permet entre autre de gérer :
  * -l'intérieur du domaine ;
  * -les bords du domaines ;
  * -les entités extérieurs au domaines.
  *
  */

#include <arcane/ItemGroup.h>

using namespace Arcane;

class INumericalGraph;
class INumericalModel;

class INumericalDomain
{
public:
  virtual ~INumericalDomain() {}
  
  virtual INumericalModel * getParent() const = 0;
  virtual INumericalGraph * getNumericalGraph() const = 0;
  virtual CellGroup internalCells() const = 0;
  virtual FaceGroup allFaces() const = 0;
  virtual FaceGroup boundaryFaces() const = 0;
  virtual CellGroup boundaryCells() const = 0;
  
  // Gestion des bords
  virtual Integer nbBoundary() const = 0;
  virtual FaceGroup boundary(Integer index) const = 0;
};

// Qualifie un modèle d'applicable sur un INumericalDomain
class INumericalDomainModel
{
public :
  virtual ~INumericalDomainModel() {}
  
  virtual void setNumericalDomain(const INumericalDomain * domain) = 0 ;
};

// ATTENTION
// Implémentation de INumericalDomainModel à déplacer

class NumericalDomainBaseModel : public INumericalDomainModel
{
public :
  
  NumericalDomainBaseModel() : m_domain(NULL) {}
  
  virtual ~NumericalDomainBaseModel() {}
  
  void setNumericalDomain(const INumericalDomain * domain) { m_domain = domain; }

protected :
  
  const INumericalDomain * m_domain;
};

#endif /*INUMERICALDOMAIN_H_*/
