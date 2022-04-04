// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* AlephTopology.h                                                  (C) 2010 */
/*                                                                           */
/*---------------------------------------------------------------------------*/
#ifndef ALEPH_TOPOLOGY_H
#define ALEPH_TOPOLOGY_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/aleph/AlephGlobal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IAlephTopology;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur l'environement parallèle.
 */
class ARCANE_ALEPH_EXPORT AlephTopology
: public TraceAccessor
{
 public:
  AlephTopology(AlephKernel*);
  AlephTopology(ITraceMng*, AlephKernel*, Integer, Integer);
  virtual ~AlephTopology();

 public:
  void create(Integer);
  void setRowNbElements(IntegerConstArrayView row_nb_element);
  IntegerConstArrayView ptr_low_up_array();
  IntegerConstArrayView part();
  IParallelMng* parallelMng();
  void rowRange(Integer& min_row, Integer& max_row);

 private:
  inline void checkForInit()
  {
    if (m_has_been_initialized == false)
      throw FatalErrorException("AlephTopology::create", "Has not been yet initialized!");
  }

 public:
  Integer rowLocalRange(const Integer);
  AlephKernel* kernel(void) { return m_kernel; }
  Integer nb_row_size(void)
  { /*checkForInit();*/
    return m_nb_row_size;
  }
  Integer nb_row_rank(void)
  {
    checkForInit();
    return m_nb_row_rank;
  }
  Integer gathered_nb_row(Integer i)
  {
    checkForInit();
    return m_gathered_nb_row[i];
  }
  ArrayView<Integer> gathered_nb_row_elements(void)
  {
    checkForInit();
    return m_gathered_nb_row_elements;
  }
  ArrayView<Integer> gathered_nb_setValued(void)
  {
    checkForInit();
    return m_gathered_nb_setValued;
  }
  Integer gathered_nb_setValued(Integer i)
  {
    checkForInit();
    return m_gathered_nb_setValued[i];
  }
  bool hasSetRowNbElements(void) { return m_has_set_row_nb_elements; }

 private:
  AlephKernel* m_kernel;
  Integer m_nb_row_size; // Nombre de lignes de la matrice réparties sur l'ensemble
  Integer m_nb_row_rank; // Nombre de lignes de la matrice vue de mon rang
  UniqueArray<Integer> m_gathered_nb_row; // Indices des lignes par CPU
  UniqueArray<Integer> m_gathered_nb_row_elements; // nombre d'éléments par ligne
  UniqueArray<Integer> m_gathered_nb_setValued; // nombre d'éléments setValué par CPU
  bool m_created;
  bool m_has_set_row_nb_elements;
  bool m_has_been_initialized;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
