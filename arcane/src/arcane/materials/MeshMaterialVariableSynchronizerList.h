// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizerList.h                      (C) 2000-2023 */
/*                                                                           */
/* List of variables to synchronize.                                         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZERLIST_H
#define ARCANE_MATERIALS_MESHMATERIALVARIABLESYNCHRONIZERLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/MatVarIndex.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Synchronizing a list of material variables.
 *
 * The add() method allows adding variables to synchronize.
 * You must then call apply() to perform the synchronization.
 *
 * An instance of this class can be used multiple times.
 */
class ARCANE_MATERIALS_EXPORT MeshMaterialVariableSynchronizerList
{
 private:

  class Impl;
  class SyncInfo;

 public:

  explicit MeshMaterialVariableSynchronizerList(IMeshMaterialMng* material_mng);
  ~MeshMaterialVariableSynchronizerList();

 public:

  MeshMaterialVariableSynchronizerList(const MeshMaterialVariableSynchronizerList&) = delete;
  MeshMaterialVariableSynchronizerList& operator=(const MeshMaterialVariableSynchronizerList&) = delete;
  MeshMaterialVariableSynchronizerList(const MeshMaterialVariableSynchronizerList&&) = delete;
  MeshMaterialVariableSynchronizerList& operator=(const MeshMaterialVariableSynchronizerList&&) = delete;

 public:

  //! Performs the synchronization
  void apply();

  //! Adds the variable \a var to the list of variables to synchronize
  void add(MeshMaterialVariable* var);

  //! After calling apply(), contains the size of the messages sent
  Int64 totalMessageSize() const;

  /*!
   * \brief Starts a non-blocking synchronization.
   *
   * This is only valid if IMeshMaterialMng::synchronizeVariableVersion() equals 7.
   */
  void beginSynchronize();

  /*!
   * \brief Blocks until the ongoing synchronization is finished.
   *
   * You must call beginSynchronize() before this call.
   */
  void endSynchronize();

 private:

  Impl* m_p;

 private:

  static void _beginSynchronizeMultiple(SyncInfo& sync_info);
  static void _beginSynchronizeMultiple2(SyncInfo& sync_info);
  static void _endSynchronizeMultiple2(SyncInfo& sync_info);
  void _fillSyncInfo(SyncInfo& sync_info);
  void _beginSynchronize(bool is_blocking);
  void _endSynchronize(bool is_blocking);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
