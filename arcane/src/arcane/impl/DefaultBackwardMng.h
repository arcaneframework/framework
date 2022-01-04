// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DefaultBackwardMng.h                                        (C) 2000-2016 */
/*                                                                           */
/* Implémentation par défaut d'une stratégie de retour-arrière.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MAIN_DEFAULTBACKWARDMNG_H
#define ARCANE_MAIN_DEFAULTBACKWARDMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/IBackwardMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableFilter;
class IDataReaderWriter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \Implémentation par défaut d'une stratégie de retour-arrière.
 */
class ARCANE_IMPL_EXPORT DefaultBackwardMng
: public IBackwardMng
{
 private:

  enum eSequence
  {
    //! Sauvegarde
    SEQSave ,
    //! Sauvegarde en force
    SEQForceSave ,
    //! Restauration
    SEQRestore ,
    //! Lock
    SEQLock ,
    //! Nothing
    SEQNothing
  };

 public:

  DefaultBackwardMng(ITraceMng* trace,ISubDomain* sub_domain);
  ~DefaultBackwardMng();

  void init() override {}

  void beginAction() override;

  void endAction() override;

  void setSavePeriod(Integer n) override { m_period = n; }
  Integer savePeriod() const override { return m_period; }

  void goBackward() override;

  bool isLocked() const override { return m_sequence == SEQLock; }

  bool isBackwardEnabled() const override { return m_sequence == SEQRestore; }

  void clear() override;

  virtual bool checkAndApplyRestore() override;
  virtual bool checkAndApplySave(bool is_forced) override;

 private:

  void _restore();

  void _save();

 private:

  ITraceMng* m_trace;
  ISubDomain* m_sub_domain;
  IVariableFilter* m_filter;
  IDataReaderWriter* m_data_io;

  //! Temps du dernier retour demandé
  Real m_backward_time;

  //! Période entre deux sauvegardes pour le retour-arrière
  Integer m_period;

  //! First save
  bool m_first_save;

  //! Actions authorisées ?
  bool m_action_refused;

  //! Séquence
  eSequence m_sequence;

 private:

  void _checkValidAction();
  void _checkSave(bool is_forced);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
