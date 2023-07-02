// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* BasicReaderWriter.h                                         (C) 2000-2023 */
/*                                                                           */
/* Lecture/Ecriture simple.                                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_STD_INTERNAL_BASICREADERWRITER_H
#define ARCANE_STD_INTERANL_BASICREADERWRITER_H
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

#include "arcane/std/BasicReaderWriterDatabase.h"

#include <map>
#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ISerializedData;
class IParallelMng;
class ParallelDataWriter;
class ParallelDataReader;
namespace impl
{
  class TextWriter;
}
} // namespace Arcane

namespace Arcane::impl
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class VariableDataInfo
{
 public:

  VariableDataInfo(const String& full_name, const ISerializedData* sdata);
  VariableDataInfo(const String& full_name, const XmlNode& element);

 public:

  const String& fullName() const { return m_full_name; }
  Integer nbDimension() const { return m_nb_dimension; }
  Int64 dim1Size() const { return m_dim1_size; }
  Int64 dim2Size() const { return m_dim2_size; }
  Int64 nbElement() const { return m_nb_element; }
  Int64 nbBaseElement() const { return m_nb_base_element; }
  Integer dimensionArraySize() const { return m_dimension_array_size; }
  bool isMultiSize() const { return m_is_multi_size; }
  eDataType baseDataType() const { return m_base_data_type; }
  Int64 memorySize() const { return m_memory_size; }
  const ArrayShape& shape() const { return m_shape; }
  void setFileOffset(Int64 v) { m_file_offset = v; }
  Int64 fileOffset() const { return m_file_offset; }

 public:

  void write(XmlNode element);

 private:

  void _addAttribute(XmlNode& node, const String& attr_name, Int64 value)
  {
    node.setAttrValue(attr_name, String::fromNumber(value));
  }

  void _addAttribute(XmlNode& node, const String& attr_name, const String& value)
  {
    node.setAttrValue(attr_name, value);
  }

  Integer _readInteger(const XmlNode& node, const String& attr_name)
  {
    return node.attr(attr_name, true).valueAsInteger(true);
  }

  Int64 _readInt64(const XmlNode& node, const String& attr_name)
  {
    return node.attr(attr_name, true).valueAsInt64(true);
  }

  bool _readBool(const XmlNode& node, const String& attr_name)
  {
    return node.attr(attr_name, true).valueAsBoolean(true);
  }

  String _readString(const XmlNode& node, const String& attr_name)
  {
    return node.attr(attr_name, true).value();
  }

 private:

  String m_full_name;
  Integer m_nb_dimension;
  Int64 m_dim1_size;
  Int64 m_dim2_size;
  Int64 m_nb_element;
  Int64 m_nb_base_element;
  Integer m_dimension_array_size;
  bool m_is_multi_size;
  eDataType m_base_data_type;
  Int64 m_memory_size;
  Int64 m_file_offset;
  ArrayShape m_shape;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicVariableMetaData
{
 public:

  BasicVariableMetaData(VariableMetaData* varmd)
  {
    m_full_name = varmd->fullName();
    m_item_group_name = varmd->itemGroupName();
    m_mesh_name = varmd->meshName();
    m_item_family_name = varmd->itemFamilyName();
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

  virtual ~IGenericReader() {}

 public:

  virtual void initialize(const String& path, Int32 rank) = 0;
  virtual void readData(const String& var_full_name, IData* data) = 0;
  virtual void readItemGroup(const String& group_name, Int64Array& written_unique_ids,
                             Int64Array& wanted_unique_ids) = 0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class BasicGenericReader
: public TraceAccessor
, public IGenericReader
{
 public:

  // Si 'version==-1', alors cela sera déterminé lors de
  // l'initialisation.
  BasicGenericReader(IApplication* app, Int32 version, Ref<KeyValueTextReader> text_reader);
  ~BasicGenericReader() override;

 public:

  void initialize(const String& path, Int32 rank) override;
  void readData(const String& var_full_name, IData* data) override;
  void readItemGroup(const String& group_name, Int64Array& written_unique_ids,
                     Int64Array& wanted_unique_ids) override;

 private:

  using VariableDataInfoMap = std::map<String, VariableDataInfo*>;

  IApplication* m_application;
  Ref<KeyValueTextReader> m_text_reader;
  String m_path;
  Int32 m_rank;
  Int32 m_version;
  VariableDataInfoMap m_variables_data_info;

 private:

  VariableDataInfo* _getVarInfo(const String& full_name);
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

  virtual ~IGenericWriter() {}

 public:

  virtual void initialize(const String& path, Int32 rank) = 0;
  virtual void writeData(const String& var_full_name, const ISerializedData* sdata) = 0;
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
  ~BasicGenericWriter() override;

 public:

  void initialize(const String& path, Int32 rank) override;
  void writeData(const String& var_full_name, const ISerializedData* sdata) override;
  void writeItemGroup(const String& group_full_name, SmallSpan<const Int64> written_unique_ids,
                      SmallSpan<const Int64> wanted_unique_ids) override;
  void endWrite() override;

 private:

  using VariableDataInfoMap = std::map<String, VariableDataInfo*>;

  IApplication* m_application;
  Int32 m_version;
  String m_path;
  Int32 m_rank;
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
  ~BasicReaderWriterCommon();

 protected:

  IApplication* m_application;
  IParallelMng* m_parallel_mng;
  eOpenMode m_open_mode;
  String m_path;
  Integer m_verbose_level;

 protected:

  String _getMetaDataFileName(Int32 rank);

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
/*!
 * \brief Lecture/Ecriture simple.
 */
class BasicWriter
: public BasicReaderWriterCommon
, public IDataWriter
{
 public:

  BasicWriter(IApplication* app, IParallelMng* pm, const String& path,
              eOpenMode open_mode, Integer version, bool want_parallel);
  ~BasicWriter() override;

 public:

  //! Positionne le service de compression. Doit être appelé avant initialize()
  void setDataCompressor(Ref<IDataCompressor> data_compressor)
  {
    m_data_compressor = data_compressor;
  }
  void initialize();

  void beginWrite(const VariableCollection& vars) override;
  void endWrite() override;

  void setMetaData(const String& meta_data) override;

  void write(IVariable* v, IData* data) override;

 private:

  bool m_want_parallel;
  bool m_is_gather;
  Int32 m_version;

  Ref<IDataCompressor> m_data_compressor;
  Ref<IHashAlgorithm> m_hash_algorithm;
  Ref<KeyValueTextWriter> m_text_writer;

  std::map<ItemGroup, ParallelDataWriter*> m_parallel_data_writers;
  std::set<ItemGroup> m_written_groups;

  ScopedPtrT<IGenericWriter> m_global_writer;

 private:

  void _directWriteVal(IVariable* v, IData* data);
  void _writeVal(TextWriter* writer, VariableDataInfo* data_info,
                 const ISerializedData* sdata);

  ParallelDataWriter* _getWriter(IVariable* var);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur simple.
 */
class BasicReader
: public BasicReaderWriterCommon
, public IDataReader
, public IDataReader2
{
 public:

  /*!
    \brief  Interface pour retrouver le groupe associée à une variable à partir
    de ces meta-données.
  */
  class IItemGroupFinder
  {
   public:

    virtual ~IItemGroupFinder() {}
    virtual ItemGroup getWantedGroup(VariableMetaData* vmd) = 0;
  };

 public:

  BasicReader(IApplication* app, IParallelMng* pm, Int32 forced_rank_to_read,
              const String& path, bool want_parallel);
  ~BasicReader() override;

 public:

  void beginRead(const VariableCollection& vars) override;
  void endRead() override {}
  String metaData() override;
  void read(IVariable* v, IData* data) override;

  void fillMetaData(ByteArray& bytes) override;
  void beginRead(const DataReaderInfo& infos) override;
  void read(const VariableDataReadInfo& infos) override;

 public:

  void initialize();
  void setItemGroupFinder(IItemGroupFinder* group_finder)
  {
    m_item_group_finder = group_finder;
  }

 private:

  bool m_want_parallel;
  Integer m_nb_written_part;
  Int32 m_version;

  Int32 m_first_rank_to_read;
  Int32 m_nb_rank_to_read;
  Int32 m_forced_rank_to_read;

  std::map<String, ParallelDataReader*> m_parallel_data_readers;
  UniqueArray<IGenericReader*> m_global_readers;
  IItemGroupFinder* m_item_group_finder;
  Ref<KeyValueTextReader> m_forced_rank_to_read_text_reader; //!< Lecteur pour le premier rang à lire.
  Ref<IDataCompressor> m_data_compressor;

 private:

  void _directReadVal(VariableMetaData* varmd, IData* data);

  ParallelDataReader* _getReader(VariableMetaData* varmd);
  void _setRanksToRead();
  IGenericReader* _readOwnMetaDataAndCreateReader(Int32 rank);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::impl

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
