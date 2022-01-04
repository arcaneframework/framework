// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AnyItemTester.cc                                            (C) 2000-2013 */
/*                                                                           */
/* Service du test avancé de l'outil AnyItem                 .               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/BasicUnitTest.h"
 
#include "arcane/tests/anyitem/AnyItemTester_axl.h"

#include "arcane/tests/ArcaneTestGlobal.h"
#include "arcane/Timer.h"

#include "arcane/anyitem/AnyItem.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arcane;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Service de test de AnyItem
 */
class AnyItemTester
  : public ArcaneAnyItemTesterObject
{
public:
 
  AnyItemTester(const ServiceBuildInfo& sbi)
    : ArcaneAnyItemTesterObject(sbi)
    , m_timer(sbi.subDomain(),"TestTimer",Arcane::Timer::TimerVirtual) 
    , m_timer_2(sbi.subDomain(),"TestTimer2",Arcane::Timer::TimerVirtual) {}

  ~AnyItemTester() {}
  
public:

  void initializeTest();
  void executeTest();
  
private:
  
  void _test1();
  void _test2();
  
private:
  
  Timer m_timer;
  Timer m_timer_2;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void 
AnyItemTester::
initializeTest()
{
  info() << "init";
  
  m_face_variable.fill(1.);
  m_cell_variable.fill(2.);

  m_face_variable_array.resize(3);
  m_cell_variable_array.resize(3);
  m_face_variable_array.fill(1.);
  m_cell_variable_array.fill(2.);
}

/*---------------------------------------------------------------------------*/

void
AnyItemTester::
executeTest()
{
  info() << "compute";
  _test1();
  _test2();
}

/*---------------------------------------------------------------------------*/

void 
AnyItemTester::
_test1()
{
  info() << "************************************** test 1 :";
 
  // Création d'une famille aggrégeant des groupes de maille et de face
  AnyItem::Family family;
  family << AnyItem::GroupBuilder( allFaces() ) 
         << AnyItem::GroupBuilder( allCells() );
  
  // Création d'une variable aggrégeant des variables de maille et de face
  AnyItem::Variable<Real> variable(family);
  variable[allFaces()] << m_face_variable;
  variable[allCells()] << m_cell_variable;
  
  Real value = 0;
  {
    // Enumération de la variable aggrégée via le groupe aggrégé
    Arcane::Timer::Sentry t(&m_timer);
    ENUMERATE_ANY_ITEM(iitem, family.allItems()) {
      value += variable[iitem];
    }
  }
  info() << "Aggregate iteration : Value = " << value 
         << ", Time = " << m_timer.lastActivationTime();
  
  // Création d'une variable aggrégeant des variables de maille et de face
  AnyItem::VariableArray<Real> variable_array(family);
  variable_array[allFaces()] << m_face_variable_array;
  variable_array[allCells()] << m_cell_variable_array;

  value = 0;
  {
    // Enumération de la variable aggrégée via le groupe aggrégé
    Arcane::Timer::Sentry t(&m_timer);
    ENUMERATE_ANY_ITEM(iitem, family.allItems()) {
      for(Arcane::Integer i = 0; i < 3; ++i)
        value += variable_array[iitem][i];
    }
  }
  info() << "Aggregate array iteration : Value = " << value
         << ", Time = " << m_timer.lastActivationTime();


  AnyItem::Array<Real> array(family.allItems());
  array.fill(0.);

  ENUMERATE_ANY_ITEM(iitem, family.allItems()) {
    array[iitem] += variable[iitem];
  }

  value = 0;
  {
    // Enumération de la variable aggrégée via le groupe aggrégé
    Arcane::Timer::Sentry t(&m_timer);
    ENUMERATE_ANY_ITEM(iitem, family.allItems()) {
      value += array[iitem];
    }
  }

  info() << "Aggregate iteration trought array : Value = " << value
         << ", Time = " << m_timer.lastActivationTime();

  AnyItem::Array2<Real> array2(family.allItems());
  array2.resize(3);

  ENUMERATE_ANY_ITEM(iitem, family.allItems()) {
    for(Arcane::Integer i = 0; i < 3; ++i)
      array2[iitem][i] += variable[iitem];
  }

  value = 0;
  {
    // Enumération de la variable aggrégée via le groupe aggrégé
    Arcane::Timer::Sentry t(&m_timer);
    ENUMERATE_ANY_ITEM(iitem, family.allItems()) {
      for(Arcane::Integer i = 0; i < 3; ++i)
        value += array2[iitem][i];
    }
  }

  info() << "Aggregate iteration trought array2 : Value = " << value
         << ", Time = " << m_timer.lastActivationTime();

  // La suite sert simplement pour l'étude de performance
  
  value = 0;
  {
    Arcane::Timer::Sentry t(&m_timer);
    ENUMERATE_FACE(iface, allFaces()) {
      value += m_face_variable[iface];
    }
    ENUMERATE_CELL(icell, allCells()) {
      value += m_cell_variable[icell];
    }
  }
  info() << "Arcane container : Value = " << value
         << ", Time = " << m_timer.lastActivationTime();

  value = 0;
  {
    Arcane::Timer::Sentry t(&m_timer);
    ENUMERATE_ITEM(iitem, allFaces()) {
      value += m_face_variable[iitem];
    }
    ENUMERATE_ITEM(iitem, allCells()) {
      value += m_cell_variable[iitem];
    }
  }
  info() << "Generic arcane container : Value = " << value
         << ", Time = " << m_timer.lastActivationTime();
}

/*---------------------------------------------------------------------------*/

void 
AnyItemTester::
_test2()
{
  info() << "************************************** test 2 :";

  Arcane::FaceGroup faces = allCells().innerFaceGroup();
  
  // Création d'une famille aggrégeant le groupe de maille xmin_cells et de face xmin_faces
  AnyItem::Family family;
  family << AnyItem::GroupBuilder( allFaces() ) 
         << AnyItem::GroupBuilder( allCells() );
  
  AnyItem::Variable<Real> variable(family);
  variable[allFaces()] << m_face_variable;
  variable[allCells()] << m_cell_variable;
  
  Real sum = 0;
  ENUMERATE_ANY_ITEM(iitem,  family.allItems()) {
    sum += variable[iitem];
  }
  
  AnyItem::LinkFamily link_family(family);
  link_family.reserve(allFaces().size());
  
  AnyItem::LinkVariable<Real> link_variable(link_family);
  AnyItem::LinkVariableArray<Real> link_variable_array(link_family);
  link_variable_array.resize(3);
  {
    Arcane::Timer::Sentry t(&m_timer);
    ENUMERATE_FACE(iface, allFaces()) {
      if(iface->isSubDomainBoundary()) return;
      AnyItem::LinkFamily::Link link = link_family.newLink();
      // TODO : CHECK if applied on non last created link : BUG ?
      link(allCells(),allCells()) << AnyItem::Pair(iface->backCell(),iface->frontCell());
      link_variable[link] = 1.;
      for(Arcane::Integer i = 0; i < 3; ++i)
        link_variable_array[link][i] = i+1.;
    }
  }
  info() << "Add link items = " << m_timer.lastActivationTime();
  
  Real value = 0;
  Real value_array = 0;
  {
    Arcane::Timer::Sentry t(&m_timer);
    ENUMERATE_ANY_ITEM_LINK(ilink, link_family) {
      if(ilink.index() < 10) {
        info() << "back item = [uid=" << family.item(ilink.back()).uniqueId() 
               << ",lid=" << family.item(ilink.back()).localId() << ",kind="
               << family.item(ilink.back()).kind() << "]";
        info() << "front item = [uid=" << family.item(ilink.front()).uniqueId() 
               << ",lid=" << family.item(ilink.front()).localId() << ",kind="
               << family.item(ilink.front()).kind() << "]";
      }
      value += link_variable[ilink] + variable[ilink.back()] + variable[ilink.front()];
      for(Arcane::Integer i = 0; i < 3; ++i)
        value_array += link_variable_array[ilink][i] + variable[ilink.back()] + variable[ilink.front()];
    }
  }
  info() << "enumerate links = " << m_timer.lastActivationTime() << ", value = " << value;
} 

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE_ANYITEMTESTER(AnyItemTester,AnyItemTester);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANETEST_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
