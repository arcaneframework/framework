// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReaderWriter.h                                         (C) 2000-2024 */
/*                                                                           */
/* Lecture/Ecriture simple.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_BASICREADERWRITER_H
#define ARCANE_STD_INTERNAL_BASICREADERWRITER_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/String.h"
#include "arcane/utils/ArrayShape.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/Array.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/XmlNode.h"
#include "arcane/core/VariableMetaData.h"

#include "arcane/core/IDataWriter.h"
#include "arcane/core/IDataReader.h"
#include "arcane/core/IDataReader2.h"

#include "arcane/std/internal/BasicReaderWriterDatabase.h"
#include "arcane/std/internal/VariableDataInfo.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISerializedData;
class IParallelMng;
class ParallelDataWriter;
class ParallelDataReader;
}

namespace Arcane::impl
{
class TextWriter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicVariableMetaData
{
 public:

  explicit BasicVariableMetaData(VariableMetaData* varmd)
  : m_full_name(varmd->fullName())
  , m_item_group_name(varmd->itemGroupName())
  , m_mesh_name(varmd->meshName())
  , m_item_family_name(varmd->itemFamilyName())
  {
  }

 public:

  const String& fullName() const { return m_full_name; }
  const String& itemGroupName() const { return m_item_group_name; }
  const String& meshName() const { return m_mesh_name; }
  const String& itemFamilyName() const { return m_item_family_name; }
  bool isItemVariable() const { return !m_item_family_name.null(); }

 private:

  String m_full_name;
  String m_item_group_name;
  String m_mesh_name;
  String m_item_family_name;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Lecteur générique.
 */
class IGenericReader
{
 public:

  virtual ~IGenericReader() = default;

 public:

  virtual void initialize(const String& path, Int32 rank) = 0;
  virtual void readData(const String& var_full_name, IData* data) = 0;
  virtual void readItemGroup(const String& group_name, Int64Array& written_unique_ids,
                             Int64Array& wanted_unique_ids) = 0;
  virtual String comparisonHashValue(const String& var_full_name) const = 0;
  virtual const VariableDataInfoMap& variablesDataInfoMap() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Implémentation basique de \a IGenericReader
 */
class BasicGenericReader
: public TraceAccessor
, public IGenericReader
{
 public:

  // Si 'version==-1', alors cela sera déterminé lors de
  // l'initialisation.
  BasicGenericReader(IApplication* app, Int32 version, Ref<KeyValueTextReader> text_reader);

 public:

  void initialize(const String& path, Int32 rank) override;
  void readData(const String& var_full_name, IData* data) override;
  void readItemGroup(const String& group_name, Int64Array& written_unique_ids,
                     Int64Array& wanted_unique_ids) override;
  String comparisonHashValue(const String& var_full_name) const override;
  const VariableDataInfoMap& variablesDataInfoMap() const override
  {
    return m_variables_data_info;
  }

 private:

  IApplication* m_application = nullptr;
  Ref<KeyValueTextReader> m_text_reader;
  String m_path;
  Int32 m_rank = A_NULL_RANK;
  Int32 m_version = -1;
  VariableDataInfoMap m_variables_data_info;

 private:

  Ref<VariableDataInfo> _getVarInfo(const String& full_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Écrivain générique.
 */
class IGenericWriter
{
 public:

  virtual ~IGenericWriter() = default;

 public:

  virtual void initialize(const String& path, Int32 rank) = 0;
  /*!
   * \brief Sauve une variable.
   * \param var_full_name Nom de la variable
   * \param sdata valeurs sérialisées de la variable
   * \param comparison_hash hash de comparaison (null si aucun)
   * \param is_save_values Indique si on sauvegarde les valeurs.
   */
  virtual void writeData(const String& var_full_name, const ISerializedData* sdata,
                         const String& comparison_hash,
                         bool is_save_values) = 0;
  virtual void writeItemGroup(const String& group_full_name,
                              SmallSpan<const Int64> written_unique_ids,
                              SmallSpan<const Int64> wanted_unique_ids) = 0;
  virtual void endWrite() = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicGenericWriter
: public TraceAccessor
, public IGenericWriter
{
 public:

  BasicGenericWriter(IApplication* app, Int32 version,
                     Ref<KeyValueTextWriter> text_writer);

 public:

  void initialize(const String& path, Int32 rank) override;
  void writeData(const String& var_full_name, const ISerializedData* sdata,
                 const String& compare_hash, bool is_save_values) override;
  void writeItemGroup(const String& group_full_name, SmallSpan<const Int64> written_unique_ids,
                      SmallSpan<const Int64> wanted_unique_ids) override;
  void endWrite() override;

 private:

  IApplication* m_application = nullptr;
  Int32 m_version = -1;
  String m_path;
  Int32 m_rank = A_NULL_RANK;
  Ref<KeyValueTextWriter> m_text_writer;
  VariableDataInfoMap m_variables_data_info;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicReaderWriterCommon
: public TraceAccessor
{
  // Pour accéder aux méthodes statiques
  friend class BasicGenericReader;
  friend class BasicGenericWriter;

 public:

  enum eOpenMode
  {
    OpenModeRead,
    OpenModeTruncate,
    OpenModeAppend
  };

 public:

  BasicReaderWriterCommon(IApplication* app, IParallelMng* pm,
                          const String& path, eOpenMode open_mode);

 protected:

  IApplication* m_application = nullptr;
  IParallelMng* m_parallel_mng = nullptr;
  eOpenMode m_open_mode = OpenModeRead;
  String m_path;
  Integer m_verbose_level = 0;

 protected:

  String _getMetaDataFileName(Int32 rank) const;

 protected:

  static String _getArcaneDBTag();
  static String _getOwnMetatadaFile(const String& path, Int32 rank);
  static String _getArcaneDBFile(const String& path, Int32 rank);
  static String _getBasicVariableFile(Int32 version, const String& path, Int32 rank);
  static String _getBasicGroupFile(const String& path, const String& name, Int32 rank);
  static Ref<IDataCompressor> _createDeflater(IApplication* app, const String& name);
  static Ref<IHashAlgorithm> _createHashAlgorithm(IApplication* app, const String& name);
  static void _fillUniqueIds(const ItemGroup& group, Array<Int64>& uids);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
