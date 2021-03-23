// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
#ifndef SDMBOUNDARYCONDITIONMNG_H_
#define SDMBOUNDARYCONDITIONMNG_H_

#include "NumericalModel/SubDomainModel/SDMBoundaryCondition.h"

using namespace Arcane;

template<typename Item, typename ModelProperty>
class ModelBoundaryConditionMng
{
public :
  typedef typename ModelProperty::NumericalDomain NumericalDomain;
  typedef ModelBoundaryCondition< Item,ModelProperty > BoundaryCondition;
  typedef BufferT< BoundaryCondition* > BoundaryConditionList;
  typedef typename BufferT< BoundaryCondition* >::iter BoundaryConditionIter;
  typedef typename ModelBoundaryCondition< Item,ModelProperty >::IBCOp IBoundaryConditionOp;
  typedef BufferT< IBoundaryConditionOp* > BoundaryConditionOpList;
  typedef typename BufferT< IBoundaryConditionOp* >::iter BoundaryConditionOpIter;
  
  ModelBoundaryConditionMng(NumericalDomain* domain) : m_domain(domain) {}

  virtual ~ModelBoundaryConditionMng() 
  {
    for(BoundaryConditionIter iter(m_bcs);iter.notEnd();++iter)
      delete (*iter);
  }
  
  BoundaryCondition* getBC(Integer id)
  { return m_bcs[id]; }
  
  void addNew(BoundaryCondition* new_bc,IBoundaryConditionOp* new_bc_op=NULL)
  {
    Integer id = new_bc->getBoundaryId();
    ItemGroupT<Item> boundary = m_domain->boundary(id);
    new_bc->setBoundary(boundary);
    m_bcs.add(new_bc);
    m_bc_ops.add(new_bc_op);
    
  }
  BoundaryConditionIter getBCIter()
  {
    return BoundaryConditionIter(m_bcs);
  }
  
  void initBC()
  {
    BoundaryConditionOpIter op_iter(m_bc_ops);
    BoundaryConditionIter bc_iter(m_bcs);
    while(bc_iter.notEnd())
    {
      IBoundaryConditionOp* op = (*op_iter);
      if(op)
         op->compute(*bc_iter);
      ++op_iter;
      ++bc_iter;
    }
  }
private :

  BoundaryConditionList m_bcs;

  NumericalDomain * m_domain;

  BufferT< IBoundaryConditionOp* > m_bc_ops;  
};

typedef ModelBoundaryConditionMng<Face,SubDomainModelProperty> SDMFaceBoundaryConditionMng;

#endif /*SDMBOUNDARYCONDITIONMNG_H_*/
