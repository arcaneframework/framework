// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Variable.h                                                  (C) 2000-2025 */
/*                                                                           */
/* Class managing the private part of a variable.                            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_VARIABLE_H
#define ARCANE_CORE_VARIABLE_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/IData.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableInfo;
class VariableBuildInfo;
template <typename T> class IDataTracerT;
class VariablePrivate;
class MemoryAccessInfo;
class IParallelMng;
class VariableResizeArgs;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \internal
 * \brief Instance of a variable.
 *
This class manages the data of a variable. This instance should generally
not be used by code developers.
 \todo explain better
	
A variable is characterized by:
 <ul>
 <li>its <b>name</b>,</li>
 <li>its <b>type</b>: real, integer, tensor, ...</li>,
 <li>its <b>kind</b>: scalar, array, nodal quantity, cell-center quantity ...</li>
 </ul>

A variable based on a type of mesh entity is called a mesh variable.

The variable is generally used by a module (IModule) via a reference (VariableRef).

Variables are <b>persistent</b> and their reading/writing is done
by the read() and write() methods.

 \warning This class is entirely managed by Arcane, and modules
that use it should generally only use it to retrieve information. Operations
that modify this instance (such as setItemGroup())
should only be used if the developer has a good understanding
of their functioning.

 * This class must not be copied. 
 */
class ARCANE_CORE_EXPORT Variable
: public TraceAccessor
, public IVariable
{
 protected:

  //! Creates a variable linked to the reference \a v.
  Variable(const VariableBuildInfo& v, const VariableInfo& vi);

 public:

  //! Frees resources
  ~Variable() override;

 public:

  //! Copy constructor (do not use)
  Variable(const Variable& from) = delete;
  //! Copy assignment operator (do not use)
  Variable& operator=(const Variable& from) = delete;

 public:

  ISubDomain* subDomain() override;
  IVariableMng* variableMng() const override;
  String fullName() const final;
  String name() const final;
  String itemFamilyName() const final;
  String meshName() const final;
  String itemGroupName() const final;
  int property() const override;
  void notifyReferencePropertyChanged() override;

 public:

  //! Sets the usage state of the variable
  void setUsed(bool v) override;

  //! Usage state of the variable
  bool isUsed() const override;

  bool isPartial() const override;

 public:

  void setTraceInfo(Integer, eTraceType) override {}

 public:

  void read(IDataReader* d) override;
  void write(IDataWriter* d) override;
  void notifyEndRead() override;
  void notifyBeginWrite() override;

 public:

  void addVariableRef(VariableRef* ref) override;
  void removeVariableRef(VariableRef* ref) override;
  VariableRef* firstReference() const override;
  Integer nbReference() const override;

  VariableMetaData* createMetaData() const override;
  Ref<VariableMetaData> createMetaDataRef() const override;
  void syncReferences() override;

  IMesh* mesh() const final;
  MeshHandle meshHandle() const final;
  ItemGroup itemGroup() const final;
  IItemFamily* itemFamily() const final;

  eItemKind itemKind() const override;
  Integer dimension() const override;
  Integer multiTag() const override;
  Int32 checkIfSync(Integer max_print) final;
  Int32 checkIfSameOnAllReplica(Integer max_print) final;
  Int32 checkIfSame(IDataReader* reader, Integer max_print, bool compare_ghost) final;

  eDataType dataType() const override;
  bool initialize(const ItemGroup& /*group*/, const String& /*value*/) override { return true; }

  IDataFactoryMng* dataFactoryMng() const final;
  void serialize(ISerializer* sbuffer, IDataOperation* operation) override;
  void serialize(ISerializer* sbuffer, Int32ConstArrayView ids, IDataOperation* operation) override;

  void resize(Integer n) override;
  void resizeFromGroup() override;

  void setAllocationInfo(const DataAllocationInfo& v) override;
  DataAllocationInfo allocationInfo() const override;

 public:

  IObservable* writeObservable() override;
  IObservable* readObservable() override;
  IObservable* onSizeChangedObservable() override;

 public:

  void addTag(const String& tagname, const String& tagvalue) override;
  void removeTag(const String& tagname) override;
  bool hasTag(const String& tagname) override;
  String tagValue(const String& tagname) override;

 public:

  void update() override;
  void setUpToDate() override;
  Int64 modifiedTime() override;
  void addDepend(IVariable* var, eDependType dt) override;
  void addDepend(IVariable* var, eDependType dt, const TraceInfo& tinfo) override;
  void removeDepend(IVariable* var) override;
  void setComputeFunction(IVariableComputeFunction* v) override;
  IVariableComputeFunction* computeFunction() override;
  void dependInfos(Array<VariableDependInfo>& infos) override;

  void update(Real wanted_time) override;

  void changeGroupIds(Int32ConstArrayView old_to_new_ids) override;

  IVariableInternal* _internalApi() override;

 public:

  IMemoryAccessTrace* memoryAccessTrace() const override { return nullptr; }

 protected:

  void _setProperty(int property);

  /*!
   * \brief Positions the data.
   *
   * If data is null, a fatal error is sent
   */
  void _setData(const Ref<IData>& data);

  //! Indicates if the variable data is valid
  void _setValidData(bool valid_data);

  /*!
   * \brief Indicates if the variable data is valid.
   *
   * The data is valid at the end of a call to setUsed().
   */
  bool _hasValidData() const;

 protected:

  virtual void _internalResize(const VariableResizeArgs& resize_args) = 0;

  void _checkSwapIsValid(Variable* rhs);
  // Temporary for memory release test
  bool _wantShrink() const;

  // Access via VariablePrivate for internal API
  friend class VariablePrivate;
  void _resize(const VariableResizeArgs& resize_args);

  //! Comparison of values between variables
  virtual VariableComparerResults _compareVariable(const VariableComparerArgs& compare_args) = 0;

 private:

  VariablePrivate* m_p; //!< Implementation

 private:

  void _checkSetItemFamily();
  void _checkSetItemGroup();
  void _checkSetProperty(VariableRef*);
  bool _hasReference() const;
  void _removeMeshReference();
  VariableMetaData* _createMetaData() const;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif

#include "arcane/core/VariableScalar.h"
#include "arcane/core/VariableArray.h"
