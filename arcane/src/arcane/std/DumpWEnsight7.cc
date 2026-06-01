// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DumpWEnsight7.cc                                            (C) 2000-2026 */
/*                                                                           */
/* Exporting files in Ensight7 gold format.                                  */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/StringBuilder.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/Iterator.h"
#include "arcane/utils/List.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Iostream.h"
#include "arcane/utils/Deleter.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/NotImplementedException.h"
#include "arcane/utils/CStringUtils.h"

#include "arcane/core/IDataWriter.h"
#include "arcane/core/Item.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/IVariable.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/IMeshSubMeshTransition.h"
#include "arcane/core/StdNum.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/Directory.h"
#include "arcane/core/MeshVariable.h"
#include "arcane/core/PostProcessorWriterBase.h"
#include "arcane/core/Service.h"
#include "arcane/core/SimpleProperty.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/SharedVariable.h"

#include "arcane/core/FactoryService.h"
#include "arcane/core/ServiceFactory.h"

#include "arcane/std/Ensight7PostProcessor_axl.h"
#include "arcane/std/DumpW.h"

#include <string.h>
#include <memory>
#include <unordered_map>

// TODO: Add test with partial variables

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 \brief Writes in Ensight7 format.
 
 The writing is done in Ensight case format and in ASCII.

 There are two saving mechanisms depending on whether the temporal aspect is
 used or not. The choice is made using the array #m_times.
 If it is empty, a snapshot is taken, meaning just an output of the variables.
 If it is not empty, a temporal output is performed, which contains a list of
 time instants, and in this case, a time save is performed. The variables are
 then saved at their current value, considering it to be the last protection
 of the array #m_times. To have larger files, #m_fileset_size protections are
 saved per file.

 In both cases, #m_base_directory contains the path and directory where the
 variables will be saved.
 
 The case format uses one file to describe the case (.case), one file to
 describe the geometry (.geo), and one file per variable.
 In the case of a temporal save, there is one geometry and one variable file
 per time instant. In this case, the file names are suffixed by the protection
 number. For example, for a variable 'Pressure', the save file name in the case
 of a snapshot is just 'Pressure'. In the case of a temporal save at the 4th
 protection, the name will be 'Pressure000004'.

 For more precision in the Ensight case format, refer to the user manual for
 Ensight6 or Ensight7.
 */
class DumpWEnsight7
: public DumpW
, public TraceAccessor
{
 public:

  static const int int_width = 10;
  static const int g_line_length = 80;

 public:

  /*!
   * \brief Correspondence between Ensight element type and Arcane.
   */
  struct EnsightPart
  {
   public:

    EnsightPart()
    : m_type(-1)
    , m_nb_node(0)
    {}
    EnsightPart(int type, Integer nb_node, const String& name)
    : m_type(type)
    , m_nb_node(nb_node)
    , m_name(name)
    {}
    EnsightPart(const EnsightPart& part)
    : m_type(part.m_type)
    , m_nb_node(part.m_nb_node)
    , m_name(part.m_name)
    , m_items(part.m_items)
    , m_reindex(part.m_reindex)
    {
      ;
    }

   public:

    inline int type() const { return m_type; }
    inline Integer nbNode() const { return m_nb_node; }
    inline const String& name() const { return m_name; }
    inline Array<Item>& items() { return m_items; }
    inline ConstArrayView<Item> items() const { return m_items; }
    inline bool hasReindex() const { return !m_reindex.empty(); }
    inline void setReindex(Integer* reindex)
    {
      m_reindex.resize(m_nb_node);
      for (Integer i = 0; i < m_nb_node; ++i)
        m_reindex[i] = reindex[i];
    }
    inline ConstArrayView<Integer> reindex() const { return m_reindex; }

   private:

    int m_type; //!< Arcane type of the element.
    Integer m_nb_node; //!< Number of nodes
    String m_name; //!< Ensight name of this element.
    UniqueArray<Item> m_items; //!< Entities of elements of this type
    UniqueArray<Integer> m_reindex;
  };

  /*!
   * \brief Information to share a group into elements of the same subtype
   */
  struct GroupPartInfo
  {
   public:

    //! Number of subtypes.
    Integer nbType() const { return m_parts.size(); }

   public:

    GroupPartInfo()
    : m_group(0)
    , m_part_id(0)
    {}
    GroupPartInfo(ItemGroup grp, Integer id, bool use_degenerated_hexa)
    : m_group(grp)
    , m_part_id(id)
    {
      m_general_item_types = std::make_unique<VariableItemInt32>(VariableBuildInfo{ m_group.mesh(), "GeneralItemTypesGroup" + m_group.name(), m_group.itemFamily()->name() }, m_group.itemKind());
      _init(use_degenerated_hexa);
    }
    Integer totalNbElement() const
    {
      Integer n = 0;
      for (Integer i = 0; i < nbType(); ++i)
        n += m_parts[i].items().size();
      return n;
    }
    const EnsightPart& typeInfo(Integer i) const { return m_parts[i]; }
    EnsightPart& typeInfo(Integer i) { return m_parts[i]; }
    ItemGroup group() const { return m_group; }
    Integer partId() const { return m_part_id; }
    Int32 generalItemTypeId(const Item& item) const
    {
      ARCANE_ASSERT(m_general_item_types, ("Cannot question an empty GroupPartInfo"));
      return (*(m_general_item_types))[item];
    }
    EnsightPart* getTypeInfo(int type)
    {
      auto ensight_part_element = m_parts_map.find(type);
      if (ensight_part_element != m_parts_map.end())
        return ensight_part_element->second;
      else
        return nullptr;
    }

   private:

    ItemGroup m_group; //!< Associated group
    Integer m_part_id; //!< Part number
    UniqueArray<EnsightPart> m_parts;
    bool m_is_polygonal_type_registration_done = false;
    bool m_is_polyhedral_type_registration_done = false;
    //! Variable to store the types of general items (untyped)
    std::unique_ptr<VariableItemInt32> m_general_item_types = nullptr;
    using TypeId = int;
    std::unordered_map<TypeId, EnsightPart*> m_parts_map; // used to handle large number of extra types

   private:

    void _initPartMap()
    {
      for (auto& ensight_part : m_parts) {
        m_parts_map[ensight_part.type()] = &ensight_part;
      }
    }

    void _init(bool use_degenerated_hexa)
    {
      ItemTypeMng* item_type_mng = m_group.mesh()->itemTypeMng();

      // NOTE: It is important that 'nfaced' type elements
      // and 'nsided' are contiguous because Ensight must save
      // their values together
      m_parts.reserve(ItemTypeMng::nbBasicItemType());
      m_parts.add(EnsightPart(IT_Line2, 2, "bar2")); // Bar
      m_parts.add(EnsightPart(IT_Triangle3, 3, "tria3")); // Triangle
      m_parts.add(EnsightPart(IT_Quad4, 4, "quad4")); // Quandrangle
      m_parts.add(EnsightPart(IT_Pentagon5, 5, "nsided")); // Pentagone
      m_parts.add(EnsightPart(IT_Hexagon6, 6, "nsided")); // Hexagone
      m_parts.add(EnsightPart(IT_Heptagon7, 7, "nsided")); // Heptagone
      m_parts.add(EnsightPart(IT_Octogon8, 8, "nsided")); // Octogone
      // Search for other 'polygonal' types
      for (Integer i_type = ItemTypeMng::nbBuiltInItemType(); i_type < ItemTypeMng::nbBasicItemType(); ++i_type) {
        ItemTypeInfo* type_info = item_type_mng->typeFromId(i_type);
        if (type_info->nbLocalNode() == type_info->nbLocalEdge()) { // Polygon found
          m_parts.add(EnsightPart(i_type, type_info->nbLocalNode(), "nsided"));
        }
      }
      // Add polygons handled in general polyhedral mesh: no types defined
      Int32 type_id = ItemTypeMng::nbBasicItemType();
      if (item_type_mng->hasGeneralCells(m_group.mesh())) {
        if (!m_is_polygonal_type_registration_done) {
          ENUMERATE_ITEM (iitem, m_group) {
            ItemWithNodes item = iitem->toItemWithNodes();
            if (item.nbNode() == item.itemBase().nbEdge()) { // polygon found
              (*(m_general_item_types))[item] = type_id;
              m_parts.add(EnsightPart(type_id++, item.nbNode(), "nsided"));
            }
          }
          m_is_polygonal_type_registration_done = true;
        }
      }
      m_parts.add(EnsightPart(IT_Tetraedron4, 4, "tetra4")); // Tetra
      m_parts.add(EnsightPart(IT_Pyramid5, 5, "pyramid5")); // Pyramide
      m_parts.add(EnsightPart(IT_Pentaedron6, 6, "penta6")); // Penta
      m_parts.add(EnsightPart(IT_Hexaedron8, 8, "hexa8")); // Hexa
      if (use_degenerated_hexa) {
        m_parts.add(EnsightPart(IT_HemiHexa7, 8, "hexa8")); // HemiHexa7
        {
          Integer reindex[8] = { 1, 6, 5, 0, 2, 3, 4, 0 };
          m_parts[m_parts.size() - 1].setReindex(reindex);
        }
        m_parts.add(EnsightPart(IT_HemiHexa6, 8, "hexa8")); // HemiHexa6
        {
          Integer reindex[8] = { 0, 1, 3, 5, 0, 2, 3, 4 };
          m_parts[m_parts.size() - 1].setReindex(reindex);
        }
        m_parts.add(EnsightPart(IT_HemiHexa5, 8, "hexa8")); // HemiHexa5
        {
          Integer reindex[8] = { 0, 1, 3, 4, 0, 2, 3, 4 };
          m_parts[m_parts.size() - 1].setReindex(reindex);
        }
        m_parts.add(EnsightPart(IT_AntiWedgeLeft6, 8, "hexa8")); // AntiWedgeLeft6
        {
          Integer reindex[8] = { 2, 0, 1, 2, 5, 3, 4, 4 };
          m_parts[m_parts.size() - 1].setReindex(reindex);
        }
        m_parts.add(EnsightPart(IT_AntiWedgeRight6, 8, "hexa8")); // AntiWedgeRight6
        {
          Integer reindex[8] = { 2, 0, 1, 1, 5, 3, 4, 5 };
          m_parts[m_parts.size() - 1].setReindex(reindex);
        }
        m_parts.add(EnsightPart(IT_DiTetra5, 8, "hexa8")); // DiTetra5
        {
          Integer reindex[8] = { 4, 4, 2, 3, 0, 1, 1, 3 };
          m_parts[m_parts.size() - 1].setReindex(reindex);
        }
      }
      else {
        m_parts.add(EnsightPart(IT_HemiHexa7, 7, "nfaced")); // HemiHexa7
        m_parts.add(EnsightPart(IT_HemiHexa6, 6, "nfaced")); // HemiHexa6
        m_parts.add(EnsightPart(IT_HemiHexa5, 5, "nfaced")); // HemiHexa5
        m_parts.add(EnsightPart(IT_AntiWedgeLeft6, 6, "nfaced")); // AntiWedgeLeft6
        m_parts.add(EnsightPart(IT_AntiWedgeRight6, 6, "nfaced")); // AntiWedgeRight6
        m_parts.add(EnsightPart(IT_DiTetra5, 5, "nfaced")); // DiTetra5
      }
      m_parts.add(EnsightPart(IT_Heptaedron10, 10, "nfaced")); // Wedge7
      m_parts.add(EnsightPart(IT_Octaedron12, 12, "nfaced")); // Wedge8
      m_parts.add(EnsightPart(IT_Enneedron14, 14, "nfaced")); // Wedge9
      m_parts.add(EnsightPart(IT_Decaedron16, 16, "nfaced")); // Wedge10
      // Search for other 'polyhedral' types
      for (Integer i_type = ItemTypeMng::nbBuiltInItemType(); i_type < ItemTypeMng::nbBasicItemType(); ++i_type) {
        ItemTypeInfo* type_info = item_type_mng->typeFromId(i_type);
        if (type_info->nbLocalNode() != type_info->nbLocalEdge()) { // Polyhedron found
          m_parts.add(EnsightPart(i_type, type_info->nbLocalNode(), "nfaced"));
        }
      }
      // Add polyhedra handled in general polyhedral mesh: no types defined
      if (item_type_mng->hasGeneralCells(m_group.mesh())) {
        if (!m_is_polyhedral_type_registration_done) {
          ENUMERATE_ITEM (iitem, m_group) {
            ItemWithNodes item = iitem->toItemWithNodes();
            if (item.nbNode() == 1)
              (*(m_general_item_types))[item] = IT_Vertex;
            else if (item.nbNode() == 2)
              (*(m_general_item_types))[item] = IT_Line2;
            else if (item.nbNode() != item.itemBase().nbEdge()) { // polyhedron found
              (*(m_general_item_types))[item] = type_id;
              m_parts.add(EnsightPart(type_id++, item.nbNode(), "nfaced"));
            }
          }
          m_is_polyhedral_type_registration_done = true;
        }
      }
      // if extra types are used, init a EnsightPart map to optimize GroupPartInfo fill
      if (ItemTypeMng::nbBuiltInItemType() < ItemTypeMng::nbBasicItemType() || item_type_mng->hasGeneralCells(m_group.mesh())) {
        _initPartMap();
      }
    }
  };

 public:

  //void writeFileString(ostream& o,ConstCString str);
  void writeFileString(std::ostream& o, const String& str);
  void writeFileInt(std::ostream& o, int value);
  void writeFileDouble(std::ostream& o, double value);
  Integer writeDoubleSize() const;
  Integer writeIntSize() const;
  void writeFileArray(std::ostream& o, IntegerConstArrayView value);
  bool isBinary() const { return m_is_binary; }

 public:

  /*!
   * \brief Functor for writing a variable.
 
   This is the base class of functors allowing a variable to be saved
   on the elements of a given group. Derived classes must define the
   operator() with the element identifier as the only parameter.
 
   For example, if we have a mesh group containing 3 meshes with
   IDs 2, 5, and 9, the functor will be called three times with each
   of its indices.

   \sa WriteDouble, WriteReal3, WriteArrayDouble, WriteArrayReal3
  */
  class WriteBase
  {
   public:

    explicit WriteBase(DumpWEnsight7& dw)
    : m_dw(dw)
    {}
    WriteBase(DumpWEnsight7& dw, GroupIndexTable* idx)
    : m_dw(dw)
    , m_idx(idx)
    {}
    WriteBase(const WriteBase& wb)
    : m_dw(wb.m_dw)
    , m_idx(wb.m_idx)
    {}
    virtual ~WriteBase() {}

   public:

    virtual WriteBase* clone() = 0;
    virtual void begin() { init(); }
    virtual void end() {}
    virtual void write(ConstArrayView<Item> items) = 0;

    virtual void init()
    {
      m_ofile.precision(5);
      m_ofile.flags(std::ios::scientific);
    }
    virtual void putValue(std::ostream& ofile)
    {
      ofile << m_ofile.str();
    }
    std::ostream& stream() { return m_ofile; }

   protected:

    DumpWEnsight7& m_dw;
    std::ostringstream m_ofile;
    GroupIndexTable* m_idx = nullptr;
  };

  /*!
   * \brief Functor for writing a variable of type <tt>Real</tt>
  */
  template <typename FromType>
  class WriteDouble
  : public WriteBase
  {
   public:

    WriteDouble(DumpWEnsight7& dw, ConstArrayView<FromType> ptr, GroupIndexTable* idx = nullptr)
    : WriteBase(dw, idx)
    , m_ptr(ptr)
    {}
    WriteDouble(const WriteDouble& wd)
    : WriteBase(wd)
    , m_ptr(wd.m_ptr)
    {}

    WriteBase* clone() override { return new WriteDouble(*this); }

   public:

    ConstArrayView<FromType> m_ptr;

   public:

    inline void write(Integer index)
    {
      if (m_idx) {
        int reindex = (*m_idx)[index];
        if (reindex < 0)
          ARCANE_FATAL("Invalid index");
        m_dw.writeFileDouble(m_ofile, Convert::toDouble(Convert::toReal(m_ptr[reindex])));
      }
      else
        m_dw.writeFileDouble(m_ofile, Convert::toDouble(Convert::toReal(m_ptr[index])));
    }

    void write(ConstArrayView<Item> items) override
    {
      for (Item e : items) {
        write(e.localId());
      }
    }
  };

  /*!
   * \brief Functor for writing a variable of type <tt>Real</tt>
  */
  template <typename FromType>
  class WriteArrayDouble
  : public WriteBase
  {
   public:

    WriteArrayDouble(DumpWEnsight7& dw, ConstArray2View<FromType> ptr, const Integer idim2,
                     GroupIndexTable* idx = nullptr)
    : WriteBase(dw, idx)
    , m_ptr(ptr)
    , m_idim2(idim2)
    {}
    WriteArrayDouble(const WriteArrayDouble& wd)
    : WriteBase(wd)
    , m_ptr(wd.m_ptr)
    , m_idim2(wd.m_idim2)
    {}

    WriteBase* clone() override { return new WriteArrayDouble(*this); }

   public:

    ConstArray2View<FromType> m_ptr;
    const Integer m_idim2;

   public:

    void write(Integer index)
    {
      if (m_idx) {
        int reindex = (*m_idx)[index];
        if (reindex < 0)
          ARCANE_FATAL("Invalid index");
        m_dw.writeFileDouble(m_ofile, Convert::toDouble(Convert::toReal(m_ptr[reindex][m_idim2])));
      }
      else
        m_dw.writeFileDouble(m_ofile, Convert::toDouble(Convert::toReal(m_ptr[index][m_idim2])));
    }

    void write(ConstArrayView<Item> items) override
    {
      for (Item e : items) {
        write(e.localId());
      }
    }
  };

  /*!
   * \brief Functor to write a variable of type <tt>Real2</tt>
   */

  /*!
   * \brief Functor to write a variable of type <tt>Real3</tt>
   */
  class WriteReal3
  : public WriteBase
  {
   public:

    WriteReal3(DumpWEnsight7& dw, ConstArrayView<Real3> ptr, GroupIndexTable* idx = nullptr)
    : WriteBase(dw, idx)
    , m_ptr(ptr)
    {}
    WriteReal3(const WriteReal3& wd)
    : WriteBase(wd)
    , m_ptr(wd.m_ptr)
    {}

    WriteBase* clone() override { return new WriteReal3(*this); }

   public:

    ConstArrayView<Real3> m_ptr;

   public:

    void begin() override
    {
      _init();
      xostr.precision(5);
      xostr.flags(std::ios::scientific);
      yostr.precision(5);
      yostr.flags(std::ios::scientific);
      zostr.precision(5);
      zostr.flags(std::ios::scientific);
    }

    void write(Integer index)
    {
      if (m_idx) {
        int reindex = (*m_idx)[index];
        if (reindex < 0)
          ARCANE_FATAL("Invalid index");
        m_dw.writeFileDouble(xostr, Convert::toDouble(m_ptr[reindex].x));
        m_dw.writeFileDouble(yostr, Convert::toDouble(m_ptr[reindex].y));
        m_dw.writeFileDouble(zostr, Convert::toDouble(m_ptr[reindex].z));
      }
      else {
        m_dw.writeFileDouble(xostr, Convert::toDouble(m_ptr[index].x));
        m_dw.writeFileDouble(yostr, Convert::toDouble(m_ptr[index].y));
        m_dw.writeFileDouble(zostr, Convert::toDouble(m_ptr[index].z));
      }
    }

    void write(ConstArrayView<Item> items) override
    {
      for (Item i : items) {
        write(i.localId());
      }
    }

    void end() override
    {
      m_ofile << xostr.str();
      m_ofile << yostr.str();
      m_ofile << zostr.str();
    }

   public:

    std::ostringstream xostr;
    std::ostringstream yostr;
    std::ostringstream zostr;

   private:

    void _init()
    {
      init();
    }
  };

  /*!
   * \brief Functor to write a variable of type <tt>Real3</tt>
   */
  class WriteArrayReal3
  : public WriteBase
  {
   public:

    WriteArrayReal3(DumpWEnsight7& dw, ConstArray2View<Real3> ptr,
                    Integer idim2, GroupIndexTable* idx = nullptr)
    : WriteBase(dw, idx)
    , m_ptr(ptr)
    , m_idim2(idim2)
    {}
    WriteArrayReal3(const WriteArrayReal3& wd)
    : WriteBase(wd)
    , m_ptr(wd.m_ptr)
    , m_idim2(wd.m_idim2)
    {}

    WriteBase* clone() override { return new WriteArrayReal3(*this); }

   public:

    ConstArray2View<Real3> m_ptr;
    const Integer m_idim2;

   public:

    void begin() override
    {
      _init();
      xostr.precision(5);
      xostr.flags(std::ios::scientific);
      yostr.precision(5);
      yostr.flags(std::ios::scientific);
      zostr.precision(5);
      zostr.flags(std::ios::scientific);
    }

    inline void write(Integer index)
    {
      if (m_idx) {
        int reindex = (*m_idx)[index];
        m_dw.writeFileDouble(xostr, Convert::toDouble(m_ptr[reindex][m_idim2].x));
        m_dw.writeFileDouble(yostr, Convert::toDouble(m_ptr[reindex][m_idim2].y));
        m_dw.writeFileDouble(zostr, Convert::toDouble(m_ptr[reindex][m_idim2].z));
      }
      else {
        m_dw.writeFileDouble(xostr, Convert::toDouble(m_ptr[index][m_idim2].x));
        m_dw.writeFileDouble(yostr, Convert::toDouble(m_ptr[index][m_idim2].y));
        m_dw.writeFileDouble(zostr, Convert::toDouble(m_ptr[index][m_idim2].z));
      }
    }

    void write(ConstArrayView<Item> items) override
    {
      for (Item i : items) {
        write(i.localId());
      }
    }

    void end() override
    {
      m_ofile << xostr.str();
      m_ofile << yostr.str();
      m_ofile << zostr.str();
    }

   public:

    std::ostringstream xostr;
    std::ostringstream yostr;
    std::ostringstream zostr;

   private:

    void _init()
    {
      init();
    }
  };

 public:

  DumpWEnsight7(IMesh* m, const String& filename, RealConstArrayView times,
                VariableCollection variables, ItemGroupCollection groups,
                bool is_binary, bool is_parallel, Integer fileset_size,
                bool use_degenerated_hexa, bool force_first_geometry, bool save_uids);
  ~DumpWEnsight7();

 public:

  void writeVal(IVariable&, ConstArrayView<Byte>) override {}
  void writeVal(IVariable& v, ConstArrayView<Real> a) override { _writeRealValT<Real>(v, a); }
  void writeVal(IVariable&, ConstArrayView<Real2>) override {}
  void writeVal(IVariable&, ConstArrayView<Real3>) override;
  void writeVal(IVariable& v, ConstArrayView<Int16> a) override { _writeRealValT<Int16>(v, a); }
  void writeVal(IVariable& v, ConstArrayView<Int32> a) override { _writeRealValT<Int32>(v, a); }
  void writeVal(IVariable& v, ConstArrayView<Int64> a) override { _writeRealValT<Int64>(v, a); }
  void writeVal(IVariable&, ConstArrayView<Real2x2>) override {}
  void writeVal(IVariable&, ConstArrayView<Real3x3>) override {}
  void writeVal(IVariable&, ConstArrayView<String>) override {}

  void writeVal(IVariable&, ConstArray2View<Byte>) override {}
  void writeVal(IVariable& v, ConstArray2View<Real> a) override { _writeRealValT<Real>(v, a); }
  void writeVal(IVariable&, ConstArray2View<Real2>) override {}
  void writeVal(IVariable&, ConstArray2View<Real3>) override;
  void writeVal(IVariable& v, ConstArray2View<Int16> a) override { _writeRealValT<Int16>(v, a); }
  void writeVal(IVariable& v, ConstArray2View<Int32> a) override { _writeRealValT<Int32>(v, a); }
  void writeVal(IVariable& v, ConstArray2View<Int64> a) override { _writeRealValT<Int64>(v, a); }
  void writeVal(IVariable&, ConstArray2View<Real2x2>) override {}
  void writeVal(IVariable&, ConstArray2View<Real3x3>) override {}

  void writeVal(IVariable&, ConstMultiArray2View<Byte>) override {}
  void writeVal(IVariable& v, ConstMultiArray2View<Real> a) override { _writeRealValT<Real>(v, a); }
  void writeVal(IVariable&, ConstMultiArray2View<Real2>) override {}
  void writeVal(IVariable&, ConstMultiArray2View<Real3> a) override;
  void writeVal(IVariable& v, ConstMultiArray2View<Int16> a) override { _writeRealValT<Int16>(v, a); }
  void writeVal(IVariable& v, ConstMultiArray2View<Int32> a) override { _writeRealValT<Int32>(v, a); }
  void writeVal(IVariable& v, ConstMultiArray2View<Int64> a) override { _writeRealValT<Int64>(v, a); }
  void writeVal(IVariable&, ConstMultiArray2View<Real2x2>) override {}
  void writeVal(IVariable&, ConstMultiArray2View<Real3x3>) override {}

  void beginWrite() override;
  void endWrite() override;
  void setMetaData(const String&) override {};

  bool isParallelOutput() const { return m_is_parallel_output; }
  bool isMasterProcessor() const { return m_is_master; }
  bool isOneFilePerTime() const { return m_fileset_size == 0; }
  Int32 rank() const { return m_parallel_mng->commRank(); }
  IParallelMng* parallelMng() const { return m_parallel_mng; }

 public:
 private:

  typedef UniqueArray<GroupPartInfo*> GroupPartInfoList;

 private:

  IMesh* m_mesh; //!< Mesh
  IParallelMng* m_parallel_mng; //!< Parallelism manager
  Directory m_base_directory; //!< Storage directory name
  Directory m_part_directory; //!< Storage directory for the current iteration
  std::ofstream m_case_file; //!< File describing the case
  std::ostringstream m_case_file_variables; //! description of saved variables
  RealUniqueArray m_times; //!< List of time instants
  VariableList m_save_variables; //!< List of variables to export
  ItemGroupList m_save_groups; //!< List of groups to export
  GroupPartInfoList m_parts; //! List of parts
  bool m_is_binary; //!< \a true if file is in binary format
  bool m_is_master; //!< \a true if the processor manages the output
  /*! \a true if outputs are parallel with a proc collecting the outputs
   * from others */
  bool m_is_parallel_output;
  bool m_use_degenerated_hexa;
  bool m_force_first_geometry;
  bool m_save_uids;
  //! Total number of mesh elements across all groups to be saved
  Integer m_total_nb_element;
  Integer m_total_nb_group; //!< Number of groups to save (== number of parts)
  /*!< \brief Number of elements in a timeset
   * When saving multiple time instants in one file, a maximum of \a m_fileset_size time instants are placed in one file, and
   * when this number is reached, another file is used. */
  Integer m_fileset_size;

 private:
 private:

  //! Maximum number of digits to indicate the protection number.
  static const Integer m_max_prots_digit = 6;

 private:
 private:

  bool _isValidVariable(IVariable&) const;
  void _createCaseFile();
  void _buildFileName(const String& varname, String& filename);
  void _buildPartDirectory();
  void _writeWildcardFilename(std::ostream& ofile, const String& filename, char joker = '*');
  int _fileOuttype() const;
  void _writeFileHeader(std::ostream& o, bool write_c_binary);
  bool _isNewBlocFile() const;

  void _computeGroupParts(ItemGroupList list_group, Integer& partid);
  void _saveGroup(std::ostream& ofile, const GroupPartInfo& ensight_grp,
                  ConstArrayView<Integer> nodes_index, WriteBase& wf);
  void _saveVariableOnGroup(std::ostream& ofile, const GroupPartInfo& ensight_grp,
                            WriteBase& from_func);
  bool _isSameKindOfGroup(const ItemGroup& group, eItemKind item_kind);

  //! Template for writing variable as a real variable
  template <typename T>
  void _writeRealValT(IVariable& v, ConstArrayView<T> a);
  //! Template for writing array variable as an array real variable
  template <typename T>
  void _writeRealValT(IVariable& v, ConstArray2View<T> a);
  //! Template for writing array variable as an array real variable
  template <typename T>
  void _writeRealValT(IVariable& v, ConstMultiArray2View<T> a);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

extern "C++" DumpW*
createEnsight7(ISubDomain* m, const String& f,
               ConstArrayView<Real> times,
               VariableCollection variables,
               ItemGroupCollection groups,
               bool is_binary, bool is_parallel_output, Integer fileset_size,
               bool use_degenerated_hexa, bool force_first_geometry, bool save_uids)
{
  return new DumpWEnsight7(m->defaultMesh(), f, times, variables, groups, is_binary,
                           is_parallel_output, fileset_size, use_degenerated_hexa,
                           force_first_geometry, save_uids);
}

extern "C++" DumpW*
createEnsight7(IMesh* m, const String& f,
               ConstArrayView<Real> times,
               VariableCollection variables,
               ItemGroupCollection groups,
               bool is_binary, bool is_parallel_output, Integer fileset_size,
               bool use_degenerated_hexa, bool force_first_geometry, bool save_uids)
{
  return new DumpWEnsight7(m, f, times, variables, groups, is_binary,
                           is_parallel_output, fileset_size, use_degenerated_hexa,
                           force_first_geometry, save_uids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DumpWEnsight7::
DumpWEnsight7(IMesh* mesh, const String& filename, ConstArrayView<Real> times,
              VariableCollection variables, ItemGroupCollection groups,
              bool is_binary, bool is_parallel_output, Integer fileset_size,
              bool use_degenerated_hexa, bool force_first_geometry, bool save_uids)
: TraceAccessor(mesh->traceMng())
, m_mesh(mesh)
, m_parallel_mng(mesh->parallelMng())
, m_base_directory(filename)
, m_times(times)
, m_save_variables(variables.clone())
, m_save_groups(groups.enumerator())
, m_is_binary(is_binary)
//	, m_is_binary(false)
, m_is_master(true)
, m_is_parallel_output(is_parallel_output)
, m_use_degenerated_hexa(use_degenerated_hexa)
, m_force_first_geometry(force_first_geometry)
, m_save_uids(save_uids)
, m_total_nb_element(0)
, m_total_nb_group(0)
, m_fileset_size(fileset_size)
{
  //m_is_binary = true;
  //m_fileset_size = 0;

  if (m_times.empty()) {
    m_times.resize(1);
    m_times[0] = 0.;
  }

  //_createCaseFile();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DumpWEnsight7::
~DumpWEnsight7()
{
  std::for_each(std::begin(m_parts), std::end(m_parts), Deleter());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Retrieves a group of a given type and its information.
 *
 * \relates DumpWEnsight7
 *
 * The template parameter is either a DumpWEnsight7::CellGroup,
 * or a DumpWEnsight7::FaceGroup.
 *
 * First, iterate through the list of groups (\a list_group) and retrieve
 * those of type T::GroupType. Add these groups to the list \a grp_list and
 * assign them a number \a partid. Note that \a partid is passed by
 * reference and is incremented in this function. This identifier corresponds
 * to the part number (part) in ensight.
 *
 * Next, determine for each of its groups the number of elements of
 * each Ensight subtype. For example, for a mesh group, determine
 * the number of \e hexa8, the number of \e pyramid5, ...
 */
void DumpWEnsight7::
_computeGroupParts(ItemGroupList list_group, Integer& partid)
{
  for (ItemGroupList::Enumerator i(list_group); ++i;) {
    ItemGroup grp(*i);
    if (grp.null()) // The group is not of the desired type.
      continue;
    if (!grp.isOwn()) // If possible, take the own element group
      grp = grp.own();
    // Since Ensight 7.6, empty groups are allowed
    // and they are necessary when the mesh evolves over time
    //if (grp.empty()) // The group is empty
    //continue;
    GroupPartInfo* gpi = new GroupPartInfo(grp, partid, m_use_degenerated_hexa);
    m_parts.add(gpi);
    ++partid;

    GroupPartInfo& current_grp = *gpi;
    // Now we must determine how many elements of each type
    // ensight (tria3, hexa8, ...) are in the group.
    // two versions: if extra item types are added switch to an optimized version
    // (a large amount of types may be added, equal to the number of elements)
    if (ItemTypeMng::nbBuiltInItemType() == ItemTypeMng::nbBasicItemType()) // no extra type
    {
      debug(Trace::High) << "Using standard group part building algo";
      if (!grp.mesh()->itemTypeMng()->hasGeneralCells(grp.mesh())) { // classical types
        for (Integer z = 0; z < current_grp.nbType(); ++z) {
          EnsightPart& type_info = current_grp.typeInfo(z);
          Array<Item>& items = type_info.items();
          Integer nb_of_type = 0;
          Integer type_to_seek = type_info.type();
          ENUMERATE_ITEM (i2, grp) {
            const Item& e = *i2;
            if (e.type() == type_to_seek)
              ++nb_of_type;
          }
          items.resize(nb_of_type);
          debug(Trace::High) << "Group " << grp.name() << " has "
                             << nb_of_type << " items of type " << type_info.name();
          Integer index = 0;
          ENUMERATE_ITEM (iz, grp) {
            Item mi = *iz;
            if (mi.type() == type_to_seek) {
              ItemWithNodes e = mi.toItemWithNodes();
              items[index] = e;
              ++index;
            }
          }
        }
      }
      else { // polyhedral mesh items
        ENUMERATE_ITEM (i2, grp) {
          const Item& item = *i2;
          auto item_type = current_grp.generalItemTypeId(item);
          EnsightPart* ensight_part = current_grp.getTypeInfo(item_type);
          if (!ensight_part)
            continue;
          ItemWithNodes item_wn = item.toItemWithNodes();
          ensight_part->items().add(item_wn);
        }
      }
    }
    else // extra types are added (may have as many types as items...)
    // Propose an optimized version when many types exist (use of .format file)
    {
      debug(Trace::High) << "Using extra type group part building algo";
      // work only on face and cell groups
      auto item_kind = grp.itemKind();
      if (item_kind == IK_Cell || item_kind == IK_Face) {
        ENUMERATE_ITEM (item, grp) {
          auto item_type = item->type();
          EnsightPart* ensight_part = current_grp.getTypeInfo(item_type);
          if (!ensight_part)
            continue;
          ItemWithNodes item_wn = item->toItemWithNodes();
          ensight_part->items().add(item_wn); // few elements are added
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Saves the connectivity of elements in a group.
 
 \relates DumpWEnsight7

 Saves the connectivity of the elements in group \a ensight_grp. The difficulty
 comes from the fact that Ensight requires that elements be saved according to their
 type, that is, hexa on one side, tetra on the other, and so on.
 \a ensight_grp.m_nb_sub_part[] contains the number of elements
 for each ensight type. Therefore, we iterate through the list of elements of the group
autant de fois qu'il y a de possible Ensight types (4 for meshes,
2 for faces), and in each pass, we only save elements that are
of the correct type. It is a bit tedious, but it avoids having to manage a
list for each subtype.

 \param ofile output stream
 \param nodes_index index of each node in the Ensight coordinate array.
 \param ensight_grp group to save
 \param wf writer
*/
void DumpWEnsight7::
_saveGroup(std::ostream& ofile, const GroupPartInfo& ensight_grp,
           ConstArrayView<Integer> nodes_index, WriteBase& wf)
{
  ItemGroup igrp = ensight_grp.group();

  writeFileString(ofile, "part");
  writeFileInt(ofile, ensight_grp.partId());
  if (isParallelOutput()) {
    // In the case of parallel output, prefix the group name with the
    // CPU number.
    std::ostringstream ostr;
    ostr << igrp.name().localstr() << "_CPU" << rank();
    writeFileString(ofile, ostr.str().c_str());
  }
  else
    writeFileString(ofile, igrp.name());

  wf.putValue(ofile);

  IntegerUniqueArray array_id(256);

  for (Integer i = 0; i < ensight_grp.nbType(); ++i) {
    const EnsightPart& type_info = ensight_grp.typeInfo(i);
    ConstArrayView<Item> items = type_info.items();
    Integer nb_sub_part = items.size();
    if (nb_sub_part == 0)
      continue;

    String type_name = type_info.name();
    writeFileString(ofile, type_name);

    writeFileInt(ofile, nb_sub_part);

#if 0
    // Save the uniqueId() of the entities
    for( ConstArrayView<Item>::const_iter i(items); i(); ++i ){
      Item mi = *i;
      writeFileInt(ofile,mi.uniqueId()+1);
    }
#endif

    if (type_name == "nfaced") {
      // Special handling for non-standard meshes
      // Since our faces are not oriented relative to a
      // mesh, we use the local connectivity of each face.
      // 1. Save the number of faces for each element
      {
        if (!m_mesh->itemTypeMng()->hasGeneralCells(m_mesh)) {
          // All elements have the same number of faces
          Item mi = *items.data();
          Cell cell = mi.toCell();
          Integer nb_face = cell.nbFace();
          for (Integer z = 0; z < nb_sub_part; ++z)
            writeFileInt(ofile, nb_face);
        }
        else { // mesh has general items
          // All items do not have the same face number
          for (Item mi : items) {
            writeFileInt(ofile, mi.toCell().nbFace());
          }
        }
      }
      // 2. Save for each element, the number of nodes for each
      // of these faces
      if (!m_mesh->itemTypeMng()->hasGeneralCells(m_mesh)) {
        for (Item mi : items) {
          const ItemTypeInfo* item_info = mi.typeInfo();
          Integer nb_face = item_info->nbLocalFace();
          for (Integer z = 0; z < nb_face; ++z)
            writeFileInt(ofile, item_info->localFace(z).nbNode());
        }
      }
      else { // mesh has general items
        for (Item mi : items) {
          Cell cell = mi.toCell();
          for (Face face : cell.faces()) {
            writeFileInt(ofile, face.nbNode());
          }
        }
      }
      // 3. Save for each face of each element the list of these
      // nodes
      if (!m_mesh->itemTypeMng()->hasGeneralCells(m_mesh)) {
        for (Item item : items) {
          Cell cell(item.toCell());
          const ItemTypeInfo* item_info = cell.typeInfo();
          //Cell cell = mi.toCell();
          Integer nb_face = item_info->nbLocalFace();
          for (Integer z = 0; z < nb_face; ++z) {
            const ItemTypeInfo::LocalFace& local_face = item_info->localFace(z);
            Integer nb_node = local_face.nbNode();
            array_id.resize(nb_node);
            for (Integer y = 0; y < nb_node; ++y) {
              // Node direction inversion
              // A priori, there is a bug in Ensight (7.4.1(g)) concerning the
              // intersections of this type of elements with Ensight
              // objects (plane, sphere, ...). Regardless of the orientation
              // of the retained faces, the behavior is not correct.
              array_id[y] = nodes_index[cell.node(local_face.node(y)).localId()];
            }
            writeFileArray(ofile, array_id);
          }
        }
      }
      else { // mesh has general items
        for (Item item : items) {
          Cell cell = item.toCell();
          Integer nb_face = cell.nbFace();
          for (Integer z = 0; z < nb_face; ++z) {
            const Face local_face = cell.face(z);
            Integer nb_node = local_face.nbNode();
            array_id.resize(nb_node);
            for (Integer y = 0; y < nb_node; ++y) {
              array_id[y] = nodes_index[local_face.node(y).localId()];
            }
            writeFileArray(ofile, array_id);
          }
        }
      }
    }
    else if (type_name == "nsided") {
      // Special handling for faces with more than 4 nodes:

      // 1. Save for each element, the number of its nodes
      //cerr << "** NSIDED ELEMENT\n";
      for (Item mi : items) {
        ItemWithNodes e = mi.toItemWithNodes();
        Integer nb_node = e.nbNode();
        writeFileInt(ofile, nb_node);
      }
      // 2. Save for each element the list of its nodes
      for (Item mi : items) {
        ItemWithNodes e = mi.toItemWithNodes();
        Integer nb_node = e.nbNode();
        array_id.resize(nb_node);
        for (Integer z = 0; z < nb_node; ++z) {
          array_id[z] = nodes_index[e.node(z).localId()];
        }
        writeFileArray(ofile, array_id);
      }
    }
    else {
      // General case

      Integer nb_node = type_info.nbNode();
      array_id.resize(nb_node);

      // Save the elements.

      // Save the connectivity of the elements
      if (type_info.hasReindex()) {
        ConstArrayView<Integer> reindex = type_info.reindex();
        for (Item mi : items) {
          ItemWithNodes e = mi.toItemWithNodes();
          for (Integer j = 0; j < nb_node; ++j) {
            array_id[j] = nodes_index[e.node(reindex[j]).localId()];
          }
          writeFileArray(ofile, array_id);
        }
      }
      else { // no reindex
        for (Item mi : items) {
          ItemWithNodes e = mi.toItemWithNodes();
          for (Integer j = 0; j < nb_node; ++j) {
            array_id[j] = nodes_index[e.node(j).localId()];
          }
          writeFileArray(ofile, array_id);
        }
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Applies a functor to the elements of a group.
 *
 * \relates DumpWEnsight7
 *
 * Applies the functor \a from_func to the group \a ensight_grp.
 *
 * The functor must derive from WriteBase and is used to save a variable of a
 * given type. In Ensight, variables are saved for each part and each element
 * type (hexa8, ...). The operation is similar to saving the geometry (see the
 * function _ensightSaveGroup()). Therefore, the same procedure must be followed:
 * iterate over the group as many times as there are element types, and in one
 * iteration, only save the variables for a given element type.
 *
 * \warning This function will not support the processing of variables defined
 * on a group other than the set of elements.
 *
 * \sa WriteFunctor
 *
 * \param ofile output stream
 * \param ensight_grp group
 * \param from_func functor
 */
void DumpWEnsight7::
_saveVariableOnGroup(std::ostream& ofile, const GroupPartInfo& ensight_grp,
                     WriteBase& from_func)
{
  ItemGroup igrp = ensight_grp.group();

  writeFileString(ofile, "part");
  writeFileInt(ofile, ensight_grp.partId());

  // Elements of the same type (especially 'nfaced', 'haxa8', or 'nsided')
  // must be written in a single block (thus unified type label).
  // Furthermore, vector data must be interleaved.
  String last_type_name;
  ScopedPtrT<WriteBase> func;
  for (Integer i = 0; i < ensight_grp.nbType(); ++i) {
    const EnsightPart& type_info = ensight_grp.typeInfo(i);
    ConstArrayView<Item> items(type_info.items());
    Integer nb_sub_part = items.size();
    if (nb_sub_part == 0)
      continue;
    String type_name = type_info.name();
    if (type_name != last_type_name) {
      // Close the previous write
      if (func.get()) {
        func->end();
        func->putValue(ofile);
      }
      last_type_name = type_name;
      writeFileString(ofile, type_name);
      func = from_func.clone();
      func->begin();
    }

    // Save the variables.
    func->write(items);
  }

  // Final closure
  if (func.get()) {
    func->end();
    func->putValue(ofile);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DumpWEnsight7::
writeFileInt(std::ostream& o, int value)
{
  if (m_is_binary) {
    o.write((const char*)&value, sizeof(int));
  }
  else {
    o.width(DumpWEnsight7::int_width);
    o << value;
    o << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer DumpWEnsight7::
writeIntSize() const
{
  if (m_is_binary)
    return sizeof(int);
  return DumpWEnsight7::int_width + 1;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DumpWEnsight7::
writeFileDouble(std::ostream& o, double value)
{
  if (m_is_binary) {
    float fvalue = (float)(value);
    o.write((const char*)&fvalue, sizeof(float));
  }
  else {
    o.width(12);
    o.precision(5);
    o.flags(std::ios::scientific);
    o << value;
    o << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer DumpWEnsight7::
writeDoubleSize() const
{
  Integer float_size = (Integer)(sizeof(float));
  return (m_is_binary) ? float_size : (12 + 1);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DumpWEnsight7::
writeFileArray(std::ostream& o, IntegerConstArrayView value)
{
  if (m_is_binary) {
    o.write((const char*)value.data(), sizeof(Integer) * value.size());
  }
  else {
    for (Integer i = 0, s = value.size(); i < s; ++i) {
      o.width(DumpWEnsight7::int_width);
      o << value[i];
    }
    o << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DumpWEnsight7::
writeFileString(std::ostream& o, const String& str)
{
  if (m_is_binary) {
    char buf[g_line_length];
    for (int i = 0; i < g_line_length; ++i)
      buf[i] = '\0';
    CStringUtils::copyn(buf, str.localstr(), g_line_length);
    buf[g_line_length - 1] = '\0';
    o.write(buf, g_line_length);
  }
  else {
    o << str << '\n';
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DumpWEnsight7::
_writeFileHeader(std::ostream& o, bool write_c_binary)
{
  if (m_is_master) {
    if (_isNewBlocFile() && m_is_binary && write_c_binary)
      writeFileString(o, "C Binary");
    if (m_fileset_size != 0) {
      std::ostringstream ostr;
      ostr << "BEGIN TIME STEP "
           << "# " << m_times.size();
      writeFileString(o, ostr.str().c_str());
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Abstraction of an output file for ensight.
 *
 * In the sequential case, it is actually a file. In the parallel case, it
 * can be a file or a memory stream depending on whether it is a master or
 * slave processor.
 */
class DumpWEnsight7OutFile
{
 public:

  DumpWEnsight7OutFile(DumpWEnsight7& dw, const String& filename, int outtype)
  : m_dw(dw)
  , m_filename(filename)
  , m_is_master(dw.isMasterProcessor())
  , m_is_parallel_output(dw.isParallelOutput())
  , m_stream(0)
  , m_filestream(0)
  {
    if (m_is_master) {
      m_filestream = new std::ofstream(filename.localstr(), (std::ios_base::openmode)outtype);
      m_stream = &m_strstream;
      if (!(*m_filestream))
        m_dw.warning() << "Unable to open file " << filename;
    }
    else {
      // TODO attention to memory leaks.
      m_stream = &m_strstream;
    }
    ARCANE_CHECK_PTR(m_stream);
  }
  ~DumpWEnsight7OutFile()
  {
    delete m_filestream;
  }

 public:

  void syncFile()
  {
    IParallelMng* parallel_mng = m_dw.parallelMng();
    if (m_is_master) {
      ARCANE_CHECK_PTR(m_filestream);
      ARCANE_CHECK_PTR(m_strstream);
      //Integer pos = m_strstream->tellp();
      (*m_filestream) << m_strstream.str();
    }
    if (m_is_parallel_output) {
      if (m_is_master) {
        ARCANE_CHECK_PTR(m_filestream);

        // The master iterates through all processors and requests from each
        // the size of each save, and if it is not zero,
        // receives the message.
        Integer nb_proc = parallel_mng->commSize();
        UniqueArray<int> len_array(1);
        UniqueArray<Byte> str_array;
        for (Integer i = 1; i < nb_proc; ++i) {
          m_dw.debug(Trace::High) << "Waiting for length of processor " << i;
          parallel_mng->recv(len_array, i);
          Integer len = len_array[0];
          m_dw.debug(Trace::High) << "Length of processor " << i << " : " << len;
          if (len != 0) {
            str_array.resize(len);
            m_dw.debug(Trace::High) << "Waiting for receving geom of processor " << i;
            parallel_mng->recv(str_array, i);
            m_dw.debug(Trace::High) << "Receving geom of processor " << i;
            m_filestream->write((const char*)str_array.data(), str_array.size());
          }
        }
      }
      else {
        ARCANE_CHECK_PTR(m_strstream);
        // A slave sends to the master processor (currently processor 0)
        // the size of its data and then the data itself.
        // It is important to take the length of the string returned by m_strstream
        // and not just c_str() because the stream might contain information
        // in binary format, and we would stop at the first zero.
        UniqueArray<int> len_array(1);
        std::string str = m_strstream.str();
        Integer len = arcaneCheckArraySize(str.length());
        UniqueArray<Byte> bytes(len);
        {
          // Copy the string \a str into bytes
          Integer index = 0;
          for (ConstIterT<std::string> i(str); i(); ++i, ++index)
            bytes[index] = *i;
        }
        m_dw.debug(Trace::High) << "Not a master. " << m_filename << " size = " << len;
        //Integer len = m_strstream->tellp();
        //Integer len = str.length();
        len_array[0] = len;
        m_dw.debug(Trace::High) << "Sending length for processor 0";
        parallel_mng->send(len_array, 0);
        if (len != 0) {
          //UniqueArray<char> str_array(len);
          //std::string s = m_strstream->str();
          //platform::stdMemcpy(str_array.begin(),s.c_str(),s.length());
          m_dw.debug(Trace::High) << "Sending data for processor 0";
          parallel_mng->send(bytes, 0); //str.c_str())__array,0);
          //parallel_mng->send(str_array,0);
        }
      }
    }
    if (m_is_master)
      if (!m_dw.isOneFilePerTime())
        m_dw.writeFileString(*m_filestream, "END TIME STEP");
  }

 public:

  std::ostream& operator()() { return *m_stream; }

 private:

  DumpWEnsight7& m_dw;
  String m_filename;
  bool m_is_master;
  bool m_is_parallel_output;
  std::ostream* m_stream;
  std::ostringstream m_strstream;
  std::ofstream* m_filestream;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DumpWEnsight7::
beginWrite()
{
  String buf = m_base_directory.file("ensight.case");

  IParallelMng* parallel_mng = m_parallel_mng;
  bool is_parallel = parallel_mng->isParallel();
  m_is_master = true;
  if (is_parallel && m_is_parallel_output)
    m_is_master = parallel_mng->commRank() == 0;

  // Determines and creates the directory where the outputs will be written
  // for this iteration.
  _buildPartDirectory();

  IMesh* mesh = m_mesh;

  // Retrieves the list of groups and assigns each a unique identifier
  // unique for Ensight (part). The identifier number starts at 2, number 1
  // is for the node list
  {

    ItemGroupList list_group;
    // If the list of groups to save is empty, save all groups.
    // Otherwise, save only the specified groups.
    Integer nb_group = m_save_groups.count();
    if (nb_group == 0)
      list_group.clone(mesh->groups());
    else
      list_group.clone(m_save_groups);
    // Looks at the list of partial variables, and adds their group to the list
    // of groups to save.
    for (VariableList::Enumerator ivar(m_save_variables); ++ivar;) {
      IVariable* var = *ivar;
      if (!var->isPartial())
        continue;
      ItemGroup var_group = var->itemGroup();
      if (!list_group.contains(var_group))
        list_group.add(var_group);
    }
    Integer partid = 1;
    if (m_is_parallel_output)
      partid += parallel_mng->commRank() * list_group.count();

    _computeGroupParts(list_group, partid);

    // Retrieves the total number of elements in the groups.
    Integer total_nb_element = 0;
    m_total_nb_group = 0;
    for (auto i : m_parts) {
      total_nb_element += i->totalNbElement();
      ++m_total_nb_group;
    }

    m_total_nb_element = total_nb_element;
    debug() << "Total nb element " << m_total_nb_element << " and group " << m_total_nb_group;
    debug() << "Add nodes        " << mesh->nbNode();
  }

  // Saves the geometry in Ensight7 gold format
  if (!(m_times.size() > 1 && m_force_first_geometry)) {
    String filename;
    _buildFileName("ensight.geo", filename);

    DumpWEnsight7OutFile dw_ofile(*this, filename, _fileOuttype());

    _writeFileHeader(dw_ofile(), true);

    if (m_is_master) {
      writeFileString(dw_ofile(), "Output ensight test");
      writeFileString(dw_ofile(), "File description");
      writeFileString(dw_ofile(), "node id assign");
      writeFileString(dw_ofile(), "element id assign");
    }

    IMesh* mesh = m_mesh;
    NodeGroup all_nodes = mesh->allNodes();

    // This array is used for each mesh entity, face, or edge to
    // reference its nodes relative to the coordinate array
    // used by Ensight. The first element of this array has index 1
    UniqueArray<Integer> all_nodes_index(mesh->itemFamily(IK_Node)->maxLocalId());
    all_nodes_index.fill(0);

    UniqueArray<Real3> coords_backup;
    ConstArrayView<Real3> coords_array;
    if (mesh->parentMesh()) {
      SharedVariableNodeReal3 nodes_coords(mesh->sharedNodesCoordinates());
      coords_backup.resize(mesh->nodeFamily()->maxLocalId());
      ENUMERATE_NODE (i_item, all_nodes) {
        coords_backup[i_item.localId()] = nodes_coords[i_item];
      }
      coords_array = coords_backup.view();
    }
    else {
      VariableNodeReal3& nodes_coords(mesh->toPrimaryMesh()->nodesCoordinates());
      coords_array = ConstArrayView<Real3>(nodes_coords.asArray());
    }

    WriteReal3 wf(*this, coords_array);
    {
      wf.init();
      writeFileString(wf.stream(), "coordinates");
      writeFileInt(wf.stream(), all_nodes.size());
      wf.begin();

      // Stores the local indices for Ensight
      {
        Integer ensight_index = 1;
        ENUMERATE_ITEM (i_item, all_nodes) {
          const Item& item = *i_item;
          all_nodes_index[item.localId()] = ensight_index;
          ++ensight_index;
        }
      }

      // Displays the unique node numbers
#if 0
      {
        ENUMERATE_ITEM(i_item,all_nodes){
          const Item& item = *i_item;
          writeFileInt(wf.stream(),item.uniqueId()+1);
        }
      }
#endif
      // Displays the coordinates of each node
      ENUMERATE_ITEM (i_item, all_nodes) {
        const Item& item = *i_item;
        wf.write(item.localId());
      }
      wf.end();
    }

    for (const GroupPartInfo* part : m_parts)
      _saveGroup(dw_ofile(), *part, all_nodes_index, wf);

    dw_ofile.syncFile();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DumpWEnsight7::
endWrite()
{
  // Saving unique identifiers
  if (m_save_uids) {
    VariableCellReal cell_uids(VariableBuildInfo(m_mesh, "CellUid", IVariable::PPrivate | IVariable::PTemporary));
    ENUMERATE_CELL (icell, m_mesh->allCells()) {
      cell_uids[icell] = (Real)(icell->uniqueId().asInt64());
    }
    IVariable* cell_uid_var = cell_uids.variable();
    m_save_variables.add(cell_uid_var);
    cell_uid_var->setUsed(true);
    cell_uid_var->notifyBeginWrite();
    write(cell_uid_var, cell_uid_var->data());

    // GG: NOTE: I am not sure this works well.
    //TODO: remove the use of two variables
    VariableNodeInteger node_uids(VariableBuildInfo(m_mesh, "NodeUid", IVariable::PPrivate | IVariable::PTemporary));
    UniqueArray<Real> node_uids2(m_mesh->nodeFamily()->allItems().size());
    ENUMERATE_NODE (inode, m_mesh->allNodes()) {
      node_uids[inode] = inode->uniqueId().asInt32();
      node_uids2[inode.index()] = (Real)inode->uniqueId().asInt64();
    }

    IVariable* node_uid_var = node_uids.variable();
    m_save_variables.add(node_uid_var);
    node_uid_var->setUsed(true);
    _writeRealValT<Real>(*node_uid_var, node_uids2); // work around bug for node variables
  }

  String buf = m_base_directory.file("ensight.case");

  // Only a master processor generates a 'case' file
  if (m_is_master) {

    m_case_file.open(buf.localstr());

    if (!m_case_file)
      warning() << "Unable to write to file: <" << buf << "> error: " << m_case_file.rdstate();

    debug() << "** Exporting data to " << m_base_directory.path();
    m_case_file << "FORMAT\ntype: ensight gold\n";

    m_case_file << "\nGEOMETRY\n";
    m_case_file << "model: 1";
    if (m_fileset_size != 0)
      m_case_file << " 1 ";
    if (m_force_first_geometry)
      _writeWildcardFilename(m_case_file, "ensight.geo", '0');
    else
      _writeWildcardFilename(m_case_file, "ensight.geo", '*');
    //m_case_file << " [change_coords_only]\n";
    m_case_file << "\n";

    m_case_file << "\nVARIABLE\n";
    m_case_file << m_case_file_variables.str();

    // Saving in time. For paraview, it must be placed after the variables
    m_case_file << "\nTIME\n";
    m_case_file << "time set:              1\n";
    m_case_file << "number of steps:       " << m_times.size() << '\n';
    m_case_file << "filename start number: 0\n";
    m_case_file << "filename increment:    1\n";
    m_case_file << "time values:\n";
    // Be careful not to exceed 79 characters per line
    // allowed by Ensight. To be sure, we only write one time step
    // per line. The times are saved with the maximum number of significant digits
    // because Ensight does not like two times being equal.
    std::streamsize old_precision = m_case_file.precision(FloatInfo<Real>::maxDigit());
    for (Integer i = 0, is = m_times.size(); i < is; ++i)
      m_case_file << m_times[i] << '\n';
    m_case_file << '\n';
    m_case_file.precision(old_precision);

    if (m_fileset_size != 0) {
      m_case_file << "FILE\n";
      m_case_file << "file set: 1\n";
      if (m_fileset_size != 0) {
        Integer nb_timeset = (m_times.size() / m_fileset_size);
        for (Integer i = 0; i < nb_timeset; ++i) {
          m_case_file << "filename index:  " << i << "\n";
          m_case_file << "number of steps: " << m_fileset_size << '\n';
        }
        if (nb_timeset > 0)
          m_case_file << "filename index: " << nb_timeset << "\n";
        m_case_file << "number of steps: " << m_times.size() - (nb_timeset * m_fileset_size) << '\n';
      }
      else {
        m_case_file << "number of steps: " << m_times.size() << '\n';
      }
      m_case_file << '\n';
    }
  }
  m_case_file.flush();
  m_case_file.close();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Checks the validity of the variable to be saved
 *
 * Only variables at nodes, faces, or cells are processed.
 *
 * Only variables in the #m_save_variable_list are saved. If this list is
 * empty, all are saved.
 *
 * \retval true if the variable is valid
 * \retval false otherwise
 */
bool DumpWEnsight7::
_isValidVariable(IVariable& v) const
{
  if (!v.isUsed())
    return false;
  eItemKind ik = v.itemKind();
  if (ik != IK_Cell && ik != IK_Node && ik != IK_Face)
    return false;
  // If a variable list is specified, check that the variable is in
  // this list. If this list is empty, all are automatically saved.
  Integer nb_var = m_save_variables.count();
  if (nb_var != 0) {
    return m_save_variables.contains(&v);
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Constructs the filename for a variable or mesh name.
 *
 Constructs the filename in which a variable or
 mesh named \a name will be saved. The filename is returned in \a filename.
 
 The protection number is inserted at the end of the file in the form of a number
 formatted with #m_max_prots_digit \a characters. For example, for 'Pressure'
 at iteration 4 with #m_max_prot_digit equal to 6, we will get the
 filename 'Pressure000004'.
 \todo change the filename determination and use Directory.
*/
void DumpWEnsight7::
_buildFileName(const String& name, String& filename)
{
  StringBuilder fn_builder;
  if (m_fileset_size == 0) {
    Integer current_time = m_times.size() - 1;
    fn_builder = m_base_directory.path();
    fn_builder += "/";
    fn_builder += name;
    if (m_is_master) {
      Directory dir(fn_builder.toString());
      dir.createDirectory();
    }
    fn_builder += "/";
    fn_builder += name;

    {
      OStringStream ostr;
      ostr().fill('0'); // Fill character.
      ostr().width(m_max_prots_digit);
      ostr() << current_time;
      fn_builder += ostr.str();
    }

    //fn_builder += current_time;

    //info() << " BUILD FILE NAME name=" << name << " filename=" << fn_builder;
  }
  else {
    fn_builder = m_part_directory.path();
    fn_builder += "/";
    fn_builder += name;
  }
  filename = fn_builder;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DumpWEnsight7::
_isSameKindOfGroup(const ItemGroup& group, eItemKind item_kind)
{
  eItemKind group_kind = group.itemKind();
  if (item_kind == IK_Unknown)
    return false;
  return group_kind == item_kind;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Constructs the directory where the variables will be saved.
 *
 If you want the outputs in multiple files, the directory of the current block
 is named 'bloc******'. For example, for the 4th block, it is
 'bloc000004'.

 the block number is inserted in the form of a number
 formatted with #m_max_prots_digit \a characters. 
*/
void DumpWEnsight7::
_buildPartDirectory()
{
  Integer prot_index = 0;
  bool has_prot = false;
  if (m_fileset_size != 0) {
    Integer nb_time = m_times.size();
    if (nb_time != 0)
      prot_index = (nb_time - 1) / m_fileset_size;
    has_prot = true;
  }

  if (m_is_master)
    m_base_directory.createDirectory();

  if (has_prot) {
    OStringStream ostr;
    ostr() << "bloc";
    ostr().fill('0'); // Fill character.
    ostr().width(m_max_prots_digit);
    ostr() << prot_index;
    //ostr() << '\0';
    String buf = ostr.str();
    m_part_directory = Directory(m_base_directory, buf);
    if (m_is_master)
      m_part_directory.createDirectory();
  }
  else
    m_part_directory = m_base_directory;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DumpWEnsight7::
_writeWildcardFilename(std::ostream& ofile, const String& filename, char joker)
{
  if (m_fileset_size == 0) {
    ofile << ' ' << filename << '/' << filename;
    for (Integer i = 0; i < m_max_prots_digit; ++i)
      ofile << joker;
  }
  else {
    ofile << " bloc";
    Integer nb_time = m_times.size();
    // If the number of times is less than the size of a fileset-set,
    // there is no need to put a '*'. We can directly
    // put zeros. This is also essential to prevent
    // paraview from crashing when it rereads this type of files.
    if (nb_time <= m_fileset_size)
      joker = '0';
    for (Integer i = 0; i < m_max_prots_digit; ++i)
      ofile << joker;
    ofile << "/" << filename;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool DumpWEnsight7::
_isNewBlocFile() const
{
  Integer nb_time = m_times.size();

  if (nb_time == 1)
    return true;
  if (m_fileset_size == 0)
    return true;

  Integer modulo = (nb_time - 1) % m_fileset_size;

  if (modulo == 0)
    return true;
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int DumpWEnsight7::
_fileOuttype() const
{
  return ((isBinary()) ? std::ios::binary : 0) | ((_isNewBlocFile()) ? std::ios::trunc : std::ios::app);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Saves scalar variables.
 */
template <typename T>
void DumpWEnsight7::
_writeRealValT(IVariable& v, ConstArrayView<T> ptr)
{
  debug() << "Saving variable1 " << v.name() << " ptr=" << ptr.data() << " ps=" << ptr.size();

  if (!_isValidVariable(v))
    return;

  String filename;
  _buildFileName(v.name(), filename);

  debug() << "Saving variable " << v.name() << " in " << filename;

  DumpWEnsight7OutFile dw_ofile(*this, filename, _fileOuttype());

  _writeFileHeader(dw_ofile(), false);

  if (m_is_master)
    writeFileString(dw_ofile(), v.name());
  String var_type_str;

  if (m_is_master) {
    switch (v.itemKind()) {
    case IK_Cell:
    case IK_Face:
      var_type_str = "scalar per element: ";
      break;
    case IK_Node:
      var_type_str = "scalar per node:    ";
      break;
    default:
      break;
    }

    m_case_file_variables << var_type_str;
    m_case_file_variables << "   1   ";
    if (m_fileset_size != 0)
      m_case_file_variables << " 1 ";
    m_case_file_variables << v.name();
    _writeWildcardFilename(m_case_file_variables, v.name());
    m_case_file_variables << '\n';
  }

  switch (v.itemKind()) {
  case IK_Cell:
  case IK_Face:
  case IK_Edge: {
    GroupIndexTable* idx = nullptr;

    if (v.isPartial()) {
      idx = v.itemGroup().localIdToIndex().get();
    }

    WriteDouble<T> wf(*this, ptr, idx);
    for (const GroupPartInfo* part : m_parts) {
      bool need_save = false;
      if (v.isPartial())
        need_save = v.itemGroup() == part->group();
      else
        need_save = _isSameKindOfGroup(part->group(), v.itemKind());
      if (need_save)
        _saveVariableOnGroup(dw_ofile(), *part, wf);
    }
  } break;
  case IK_Node: {
    if (v.isPartial())
      throw NotImplementedException("DumpWEnsight7::writeVal()", "partial node variable");
    WriteDouble<T> wf(*this, ptr);
    wf.begin();
    for (Integer i = 0; i < ptr.size(); ++i)
      wf.write(i);
    wf.end();
    for (const GroupPartInfo* part : m_parts) {
      writeFileString(dw_ofile(), "part");
      writeFileInt(dw_ofile(), part->partId());
      writeFileString(dw_ofile(), "coordinates");
      wf.putValue(dw_ofile());
    }
  } break;
  default:
    break;
  }

  dw_ofile.syncFile();

  //if (m_is_transient && m_is_master)
  //writeFileString(dw_ofile(),"END TIME STEP");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Saving constant dimension array variables (by row) of scalars
 */
template <typename T>
void DumpWEnsight7::
_writeRealValT(IVariable& v, ConstArray2View<T> ptr)
{
  if (!_isValidVariable(v))
    return;

  for (Integer idim2 = 0; idim2 < ptr.dim2Size(); ++idim2) {
    String vname = v.name() + String("_") + idim2;
    String filename;
    _buildFileName(vname, filename);

    debug() << "Saving variable " << v.name() << " component " << idim2 << " in " << filename;

    DumpWEnsight7OutFile dw_ofile(*this, filename, _fileOuttype());

    _writeFileHeader(dw_ofile(), false);

    if (m_is_master)
      writeFileString(dw_ofile(), vname);
    String var_type_str;

    if (m_is_master) {
      switch (v.itemKind()) {
      case IK_Cell:
      case IK_Face:
        var_type_str = "scalar per element: ";
        break;
      case IK_Node:
        var_type_str = "scalar per node:    ";
        break;
      default:
        break;
      }

      m_case_file_variables << var_type_str;
      m_case_file_variables << "   1   ";
      if (m_fileset_size != 0)
        m_case_file_variables << " 1 ";
      m_case_file_variables << vname;
      _writeWildcardFilename(m_case_file_variables, vname);
      m_case_file_variables << '\n';
    }

    switch (v.itemKind()) {
    case IK_Cell:
    case IK_Face:
    case IK_Edge: {
      GroupIndexTable* idx = nullptr;

      if (v.isPartial()) {
        idx = v.itemGroup().localIdToIndex().get();
      }

      WriteArrayDouble<T> wf(*this, ptr, idim2, idx);
      for (const GroupPartInfo* part : m_parts) {
        bool need_save = false;
        if (v.isPartial())
          need_save = v.itemGroup() == part->group();
        else
          need_save = _isSameKindOfGroup(part->group(), v.itemKind());
        if (need_save)
          _saveVariableOnGroup(dw_ofile(), *part, wf);
      }
    } break;
    case IK_Node: {
      if (v.isPartial())
        throw NotImplementedException("DumpWEnsight7::writeVal()", "partial node variable");
      WriteArrayDouble<T> wf(*this, ptr, idim2);
      wf.begin();
      for (Integer i = 0; i < ptr.dim1Size(); ++i)
        wf.write(i);
      wf.end();
      for (const GroupPartInfo* part : m_parts) {
        writeFileString(dw_ofile(), "part");
        writeFileInt(dw_ofile(), part->partId());
        writeFileString(dw_ofile(), "coordinates");
        wf.putValue(dw_ofile());
      }
    } break;
    default:
      break;
    }

    dw_ofile.syncFile();
  }

  //if (m_is_transient && m_is_master)
  //writeFileString(dw_ofile(),"END TIME STEP");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Saving non-constant dimension array variables (by row) of scalars.
 */
template <typename T>
void DumpWEnsight7::
_writeRealValT(IVariable& v, ConstMultiArray2View<T> ptr)
{
  ARCANE_UNUSED(ptr);

  if (!_isValidVariable(v))
    return;

  warning() << "Impossible to write array variable "
            << v.name() << " of non-constant size; variable saving skipped";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Saving vector variables.
 */
void DumpWEnsight7::
writeVal(IVariable& v, ConstArrayView<Real3> ptr)
{
  if (!_isValidVariable(v))
    return;

  String filename;
  _buildFileName(v.name(), filename);

  debug() << "Saving variable " << v.name() << " in " << filename;

  DumpWEnsight7OutFile dw_ofile(*this, filename, _fileOuttype());

  _writeFileHeader(dw_ofile(), false);
  if (m_is_master)
    writeFileString(dw_ofile(), v.name());

  if (m_is_master) {
    switch (v.itemKind()) {
    case IK_Cell:
    case IK_Face:
      m_case_file_variables << "vector per element: ";
      break;
    case IK_Node:
      m_case_file_variables << "vector per node:    ";
      break;
    default:
      break;
    }

    m_case_file_variables << "   1   ";
    if (m_fileset_size != 0)
      m_case_file_variables << " 1 ";
    m_case_file_variables << v.name();

    _writeWildcardFilename(m_case_file_variables, v.name());
    m_case_file_variables << '\n';
  }

  switch (v.itemKind()) {
  case IK_Cell:
  case IK_Face:
  case IK_Edge: {
    GroupIndexTable* idx = nullptr;

    if (v.isPartial()) {
      idx = v.itemGroup().localIdToIndex().get();
    }

    WriteReal3 wf(*this, ptr, idx);
    for (const GroupPartInfo* part : m_parts) {
      bool need_save = false;
      if (v.isPartial())
        need_save = v.itemGroup() == part->group();
      else
        need_save = _isSameKindOfGroup(part->group(), v.itemKind());
      if (need_save)
        _saveVariableOnGroup(dw_ofile(), *part, wf);
    }
  } break;
  case IK_Node: {
    if (v.isPartial())
      throw NotImplementedException("DumpWEnsight7::writeVal()", "partial node variable");
    WriteReal3 wf(*this, ptr);
    wf.begin();
    for (Integer i = 0; i < ptr.size(); ++i)
      wf.write(i);
    wf.end();
    for (const GroupPartInfo* part : m_parts) {
      writeFileString(dw_ofile(), "part");
      writeFileInt(dw_ofile(), part->partId());
      writeFileString(dw_ofile(), "coordinates");
      wf.putValue(dw_ofile());
    }
  } break;
  default:
    break;
  }

  dw_ofile.syncFile();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Saving vector array variables
 */
void DumpWEnsight7::
writeVal(IVariable& v, ConstArray2View<Real3> ptr)
{
  if (!_isValidVariable(v))
    return;

  for (Integer idim2 = 0; idim2 < ptr.dim2Size(); ++idim2) {
    String vname = v.name() + String("_") + idim2;
    String filename;
    _buildFileName(vname, filename);

    debug() << "Saving variable " << v.name() << " component " << idim2 << " in " << filename;

    DumpWEnsight7OutFile dw_ofile(*this, filename, _fileOuttype());

    _writeFileHeader(dw_ofile(), false);
    if (m_is_master)
      writeFileString(dw_ofile(), vname);

    if (m_is_master) {
      switch (v.itemKind()) {
      case IK_Cell:
      case IK_Face:
        m_case_file_variables << "vector per element: ";
        break;
      case IK_Node:
        m_case_file_variables << "vector per node:    ";
        break;
      default:
        break;
      }

      m_case_file_variables << "   1   ";
      if (m_fileset_size != 0)
        m_case_file_variables << " 1 ";
      m_case_file_variables << vname;

      _writeWildcardFilename(m_case_file_variables, vname);
      m_case_file_variables << '\n';
    }

    switch (v.itemKind()) {
    case IK_Cell:
    case IK_Face:
    case IK_Edge: {
      GroupIndexTable* idx = nullptr;

      if (v.isPartial()) {
        idx = v.itemGroup().localIdToIndex().get();
      }

      WriteArrayReal3 wf(*this, ptr, idim2, idx);
      for (const GroupPartInfo* part : m_parts) {
        bool need_save = false;
        if (v.isPartial())
          need_save = v.itemGroup() == part->group();
        else
          need_save = _isSameKindOfGroup(part->group(), v.itemKind());
        if (need_save)
          _saveVariableOnGroup(dw_ofile(), *part, wf);
      }
    } break;
    case IK_Node: {
      if (v.isPartial())
        throw NotImplementedException("DumpWEnsight7::writeVal()", "partial node variable");
      WriteArrayReal3 wf(*this, ptr, idim2);
      wf.begin();
      for (Integer i = 0; i < ptr.dim1Size(); ++i)
        wf.write(i);
      wf.end();
      for (const GroupPartInfo* part : m_parts) {
        writeFileString(dw_ofile(), "part");
        writeFileInt(dw_ofile(), part->partId());
        writeFileString(dw_ofile(), "coordinates");
        wf.putValue(dw_ofile());
      }
    } break;
    default:
      break;
    }

    dw_ofile.syncFile();
  }

  //if (m_is_transient && m_is_master)
  //writeFileString(dw_ofile(),"END TIME STEP");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Saving vector array variables
 */
void DumpWEnsight7::
writeVal(IVariable& v, ConstMultiArray2View<Real3> ptr)
{
  ARCANE_UNUSED(ptr);

  if (!_isValidVariable(v))
    return;

  warning() << "Impossible to write array variable " << v.name()
            << " of non-constant size; variable saving skipped";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Post-processing in Ensight7 format.
 */
class Ensight7PostProcessorService
: public PostProcessorWriterBase
{
 public:

  Ensight7PostProcessorService(const ServiceBuildInfo& sbi)
  : PostProcessorWriterBase(sbi)
  , m_mesh(sbi.mesh())
  , m_writer(nullptr)
  {
  }

  IDataWriter* dataWriter() override { return m_writer; }
  void notifyBeginWrite() override;
  void notifyEndWrite() override;
  void close() override {}

 private:

  IMesh* m_mesh;
  DumpW* m_writer;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Ensight7PostProcessorService::
notifyBeginWrite()
{
  bool is_binary = true;
  Integer fileset_size = 100;
  bool is_parallel = subDomain()->parallelMng()->isParallel();
  bool use_degenerated_hexa = true;
  bool force_first_geometry = false;
  bool save_uids = false;
  m_writer = createEnsight7(m_mesh, baseDirectoryName(), times(),
                            variables(), groups(), is_binary, is_parallel,
                            fileset_size, use_degenerated_hexa,
                            force_first_geometry, save_uids);
}

void Ensight7PostProcessorService::
notifyEndWrite()
{
  delete m_writer;
  m_writer = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Post-processing in Ensight7 format.
 */
class Ensight7PostProcessorServiceV2
: public ArcaneEnsight7PostProcessorObject
{
 public:

  typedef ArcaneEnsight7PostProcessorObject BaseType;

 public:

  explicit Ensight7PostProcessorServiceV2(const ServiceBuildInfo& sbi)
  : ArcaneEnsight7PostProcessorObject(sbi)
  , m_mesh(sbi.mesh())
  , m_writer(nullptr)
  {
  }

  IDataWriter* dataWriter() override { return m_writer; }
  void notifyBeginWrite() override;
  void notifyEndWrite() override;
  void close() override {}

  void setMesh(IMesh* mesh) override
  {
    // TODO: A supprimer car méthode obsolète et n'est pas utilisé.
    m_mesh = mesh;
  }

 private:

  IMesh* m_mesh = nullptr;
  DumpW* m_writer = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Ensight7PostProcessorServiceV2::
notifyBeginWrite()
{
  //  std::cout << " ############# IMesh = " << mesh() << " " << m_mesh << " ###########\n";

  bool is_binary = true;
  Integer fileset_size = 0;
  bool is_parallel = m_mesh->parallelMng()->isParallel();
  is_binary = options()->binaryFile();
  fileset_size = options()->filesetSize();
  bool use_degenerated_hexa = options()->useDegeneratedHexa();
  bool force_first_geometry = options()->forceFirstGeometry();
  bool save_uids = options()->saveUids();
  m_writer = createEnsight7(m_mesh, baseDirectoryName(), times(),
                            variables(), groups(), is_binary, is_parallel, fileset_size,
                            use_degenerated_hexa, force_first_geometry, save_uids);
}

void Ensight7PostProcessorServiceV2::
notifyEndWrite()
{
  delete m_writer;
  m_writer = nullptr;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SUB_DOMAIN_FACTORY(Ensight7PostProcessorService, IPostProcessorWriter,
                                   Ensight7PostProcessor);

ARCANE_REGISTER_SERVICE_ENSIGHT7POSTPROCESSOR(Ensight7PostProcessor, Ensight7PostProcessorServiceV2);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
