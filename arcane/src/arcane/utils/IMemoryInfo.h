// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IMemoryInfo.h                                               (C) 2000-2018 */
/*                                                                           */
/* Interface d'un collecteur d'informations sur l'usage mémoire.             */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_UTILS_IMEMORYINFO_H
#define ARCANE_UTILS_IMEMORYINFO_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/IFunctorWithArgument.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations sur un bloc alloué.
 */
class MemoryInfoChunk
{
 public:
  MemoryInfoChunk() : m_owner(0), m_size(0), m_alloc_id(0), m_iteration(0) {}
  MemoryInfoChunk(const void* aowner,Int64 asize,Int64 alloc_id,Integer aiteration)
  : m_owner(aowner), m_size(asize), m_alloc_id(alloc_id), m_iteration(aiteration) {}
 public:
  const void* owner() const { return m_owner; }
  Int64 size() const { return m_size; }
  Int64 allocId() const { return m_alloc_id; }
  Integer iteration() const { return m_iteration; }
  const String& stackTrace() const { return m_stack_trace; }
 public:
  void setOwner(const void* o) { m_owner = o; }
  void setStackTrace(const String& st) { m_stack_trace = st; }
 private:
  const void* m_owner;
  Int64 m_size;
  Int64 m_alloc_id;
  Integer m_iteration;
  String m_stack_trace;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un collecteur d'informations sur l'usage mémoire.
 */
class IMemoryInfo
{
 protected:
 
  IMemoryInfo() {}

 public:
	
  //! Libère les ressources
  virtual ~IMemoryInfo(){}

 public:

  //! Crée une référence sur \a owner avec les infos de trace \a trace_info
  virtual void createOwner(const void* owner,const TraceInfo& trace_info) =0;

  //! Modifie les infos de la référence \a owner
  virtual void setOwner(const void* owner,const TraceInfo& new_info) =0;

  //! Supprime la référence sur \a owner
  virtual void removeOwner(const void* owner) =0;

 public:
  
  virtual void addInfo(const void* owner,const void* ptr,Int64 size) =0;

  virtual void addInfo(const void* owner,const void* ptr,Int64 size,const void* old_ptr) =0;

  virtual void changeOwner(const void* new_owner,const void* ptr) =0;

  virtual void removeInfo(const void* owner,const void* ptr,bool can_fail=false) =0;

  virtual void printInfos(std::ostream& ostr) =0;

 public:

  virtual void beginCollect() =0;
  virtual void endCollect() =0;
  virtual bool isCollecting() const =0;

 public:

  //! Positionne le numéro de l'itération courante.
  virtual void setIteration(Integer iteration) =0;

  virtual void printAllocatedMemory(std::ostream& ostr,Integer iteration) =0;

  //! Positionne le ITraceMng pour les messages.
  virtual void setTraceMng(ITraceMng* msg) =0;

  /*!
   * \brief Indique si on active la sauvegarde de la pile d'appel.
   *
   * Si \a is_active est vrai, active la trace la pile d'appel des allocations.
   * Le tracage est conditionné à la valeur de stackTraceMinAllocSize().
   */
  virtual void setKeepStackTrace(bool is_active) =0;

  //! Indique si la sauvegarde de la pile d'appel est activée.
  virtual bool keepStackTrace() const =0;

  /*!
   * \brief Positionne la taille minimale des allocations dont on trace la pile d'appel.
   *
   * Pour toutes les allocations au dessus de \a alloc_size,
   * la pile d'appel est conservée afin de pouvoir identifier les
   * fuites mémoires. Le cout mémoire et CPU de la conservation
   * d'une pile d'appel est important et il est donc déconseillé
   * de mettre une valeur trop faible (en dessous de 1000) à \a alloc_size.
   * La conservation de la pile d'appel est désactivée si \a keepStackTrace()
   * vaut \a false.
   */
  virtual void setStackTraceMinAllocSize(Int64 alloc_size) =0;

  //! Taille minimale des allocations dont on trace la pile d'appel.
  virtual Int64 stackTraceMinAllocSize() const =0;

  //! Visiteur sur l'ensemble des blocs alloués
  virtual void visitAllocatedBlocks(IFunctorWithArgumentT<const MemoryInfoChunk&>* functor) const =0;

  virtual Int64 nbAllocation() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" ARCANE_UTILS_EXPORT IMemoryInfo*
arcaneGlobalMemoryInfo();

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  

