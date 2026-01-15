// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DumpWEnsight7.cc                                            (C) 2000-2026 */
/*                                                                           */
/* Exportations des fichiers au format Ensight7 gold.                        */
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

#include "arcane/IDataWriter.h"
#include "arcane/Item.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/IVariable.h"
#include "arcane/ISubDomain.h"
#include "arcane/IMesh.h"
#include "arcane/IMeshSubMeshTransition.h"
#include "arcane/StdNum.h"
#include "arcane/ItemGroup.h"
#include "arcane/IParallelMng.h"
#include "arcane/Directory.h"
#include "arcane/MeshVariable.h"
#include "arcane/PostProcessorWriterBase.h"
#include "arcane/Service.h"
#include "arcane/SimpleProperty.h"
#include "arcane/IItemFamily.h"
#include "arcane/VariableCollection.h"
#include "arcane/SharedVariable.h"

#include "arcane/FactoryService.h"
#include "arcane/ServiceFactory.h"

#include "arcane/std/Ensight7PostProcessor_axl.h"
#include "arcane/std/DumpW.h"

#include <string.h>
#include <memory>
#include <unordered_map>

// TODO: Ajouter test avec des variables partielles

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 \brief Ecriture au format Ensight7.
 
 L'écriture est faite au format \a case de Ensight et en ASCII.

 Il y a deux mécanismes de sauvegardes suivant qu'on utilise l'aspect
 temporel ou pas. Le choix se fait en utilisant le tableau #m_times.
 S'il est vide, on fait alors un \a snapshot, c'est à dire juste une sortie
 des variables. S'il n'est pas vide, on fait alors une sortie \a temporelle
 et il contient une liste d'instants de temps et dans
 ce cas on effectue une sauvegarde en temps. On sauvegarde alors les
 variables à leur valeur actuelle en considérant qu'il s'agit de la derniere
 protection du tableau #m_times. Afin d'avoir des fichiers plus gros, on
 sauve #m_fileset_size protections par fichier.

 Dans les deux cas, #m_base_directory contient le chemin et le répertoire où
 seront sauvegardées les variables.
 
 Le format \a case utilise un fichier pour décrire le cas
 (.case), un fichier pour décrire la géométrie (.geo) et un fichier par variable.
 Dans le cas d'une sauvegarde temporelle, il y a une géométrie et un fichier
 de variable par instant de temps. Dans ce cas, les noms des fichiers sont
 suffixés par le numéro de la protection. Par exemple, pour une variable
 \a Pression, le nom du fichier de sauvegarde dans le cas d'un snapshot est
 juste \a 'Pression'. Dans le cas d'une sauvegarde temporelle à la 4ème protection,
 le nom sera \a 'Pression000004'.

 Pour plus de précision dans le format Ensight case, se reporter à la
 notice d'utilisation de Ensight6 ou Ensight7.
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
   * \brief Correspondance entre le type des éléments ensight et Arcane.
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

    int m_type; //!< Type Arcane de l'élément.
    Integer m_nb_node; //!< Nombre de noeud
    String m_name; //!< Nom Ensight de cet élément.
    UniqueArray<Item> m_items; //!< Entités des éléments de ce type
    UniqueArray<Integer> m_reindex;
  };

  /*!
   * \brief Information pour partagé un groupe en éléments de même sous-type
   */
  struct GroupPartInfo
  {
   public:

    //! Nombre de sous-types.
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
    EnsightPart* getTypeInfo(int type) {
      auto ensight_part_element = m_parts_map.find(type);
      if (ensight_part_element != m_parts_map.end())
        return ensight_part_element->second;
      else
        return nullptr;
    }

   private:

    ItemGroup m_group; //!< Groupe associé
    Integer m_part_id; //!< Numéro de la partie
    UniqueArray<EnsightPart> m_parts;
    bool m_is_polygonal_type_registration_done = false;
    bool m_is_polyhedral_type_registration_done = false;
    //! Variable pour stocker les types des items généraux (non typés)
    std::unique_ptr<VariableItemInt32> m_general_item_types = nullptr;
    using TypeId = int;
    std::unordered_map<TypeId, EnsightPart*> m_parts_map;// used to handle large number of extra types

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

      // NOTE: il est important que les éléments de type 'nfaced'
      // et 'nsided' soient contigues car Ensight doit sauver
      // ensemble leur valeurs
      m_parts.reserve(ItemTypeMng::nbBasicItemType());
      m_parts.add(EnsightPart(IT_Line2, 2, "bar2")); // Bar
      m_parts.add(EnsightPart(IT_Triangle3, 3, "tria3")); // Triangle
      m_parts.add(EnsightPart(IT_Quad4, 4, "quad4")); // Quandrangle
      m_parts.add(EnsightPart(IT_Pentagon5, 5, "nsided")); // Pentagone
      m_parts.add(EnsightPart(IT_Hexagon6, 6, "nsided")); // Hexagone
      m_parts.add(EnsightPart(IT_Heptagon7, 7, "nsided")); // Heptagone
      m_parts.add(EnsightPart(IT_Octogon8, 8, "nsided")); // Octogone
      // Recherche des autres types 'polygone'
      for (Integer i_type = ItemTypeMng::nbBuiltInItemType(); i_type < ItemTypeMng::nbBasicItemType(); ++i_type) {
        ItemTypeInfo* type_info = item_type_mng->typeFromId(i_type);
        if (type_info->nbLocalNode() == type_info->nbLocalEdge()) { // Polygone trouvé
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
      // Recherche des autres types 'polyedre'
      for (Integer i_type = ItemTypeMng::nbBuiltInItemType(); i_type < ItemTypeMng::nbBasicItemType(); ++i_type) {
        ItemTypeInfo* type_info = item_type_mng->typeFromId(i_type);
        if (type_info->nbLocalNode() != type_info->nbLocalEdge()) { // Polyèdre trouvé
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
   * \brief Fonctor pour écrire une variable.
 
   Il s'agit de la classe de base des fonctor permettant de sauver une
   variable sur les éléments d'un groupe donné. Les classes dérivées doivent
   définir l'opérateur() avec comme unique paramètre l'identifiant de l'élément.
 
   Par exemple, si on a un groupe de mailles contenant 3 mailles d'id 2, 5 et 9,
   le fonctor sera appelé trois fois avec chacun de ses indices.

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
   * \brief Functor pour écrire une variable de type <tt>Real</tt>
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
   * \brief Functor pour écrire une variable de type <tt>Real</tt>
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
   * \brief Functor pour écrire une variable de type <tt>Real2</tt>
   */

  /*!
   * \brief Functor pour écrire une variable de type <tt>Real3</tt>
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
   * \brief Functor pour écrire une variable de type <tt>Real3</tt>
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
  void setMetaData(const String&) override{};

  bool isParallelOutput() const { return m_is_parallel_output; }
  bool isMasterProcessor() const { return m_is_master; }
  bool isOneFilePerTime() const { return m_fileset_size == 0; }
  Int32 rank() const { return m_parallel_mng->commRank(); }
  IParallelMng* parallelMng() const { return m_parallel_mng; }

 public:
 private:

  typedef UniqueArray<GroupPartInfo*> GroupPartInfoList;

 private:

  IMesh* m_mesh; //!< Maillage
  IParallelMng* m_parallel_mng; //!< Gestionnaire du parallélisme
  Directory m_base_directory; //!< Nom du répertoire de stockage
  Directory m_part_directory; //!< Répertoire de stockage de l'itération courante
  std::ofstream m_case_file; //!< Fichier décrivant le cas
  std::ostringstream m_case_file_variables; //! description des variables sauvées
  RealUniqueArray m_times; //!< Liste des instants de temps
  VariableList m_save_variables; //!< Liste des variables a exporter
  ItemGroupList m_save_groups; //!< Liste des groupes a exporter
  GroupPartInfoList m_parts; //! Liste des parties
  bool m_is_binary; //!< \a true si fichier au format binaire
  bool m_is_master; //!< \a true si le processeur dirige la sortie
  /*! \a true si les sorties sont parallèles avec un proc qui récupère les sorties
   * des autres */
  bool m_is_parallel_output;
  bool m_use_degenerated_hexa;
  bool m_force_first_geometry;
  bool m_save_uids;
  //! Nombre total d'éléments de maillage de tous les groupes à sauver
  Integer m_total_nb_element;
  Integer m_total_nb_group; //!< Nombre de groupes à sauver (== nombre de part)
  /*!< \brief Nombre d'éléments dans un timeset
   * Lorsqu'on sauve plusieurs instants de temps dans un fichier, on met
   * au maximum \a m_fileset_size instants de temps dans un fichier et
   * lorsque ce nombre est atteint, on utilise un autre fichier. */
  Integer m_fileset_size;

 private:
 private:

  //! Nombre maximum de chiffres pour indiquer le numéro de protection.
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
  //! Template for writing array variable as a array real variable
  template <typename T>
  void _writeRealValT(IVariable& v, ConstArray2View<T> a);
  //! Template for writing array variable as a array real variable
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
 * \brief Récupère un groupe d'un type donné et ses informations.
 *
 * \relates DumpWEnsight7

 Le paramètre template est soit un DumpWEnsight7::CellGroup,
 soit un DumpWEnsight7::FaceGroup.

 Tout d'abord, parcours la liste des groupes (\a list_group) et récupère
 ceux du type T::GroupType. Ajoute ces groupes à la liste \a grp_list et
 leur donne un numéro \a partid. A noter que \a partid est passé par
 référence et est incrémenté dans cette fonction. Cet identifiant correspond
 au numéro de partie (part) dans ensight.

 Ensuite, détermine pour chacun de ses groupes le nombre d'éléments de
 chaque sous type Ensight. Par exemple, pour un groupe de mailles, détermine
 le nombre de \e hexa8, le nombre de \e pyramid5, ...
*/
void DumpWEnsight7::
_computeGroupParts(ItemGroupList list_group, Integer& partid)
{
  for (ItemGroupList::Enumerator i(list_group); ++i;) {
    ItemGroup grp(*i);
    if (grp.null()) // Le groupe n'est pas du type souhaité.
      continue;
    if (!grp.isOwn()) // Si possible, prend le groupe d'éléments propres
      grp = grp.own();
    // Depuis la 7.6 de Ensight, les groupes vides sont autorisés
    // et ils sont nécessaires lorsque le maillage évolue au cours du temps
    //if (grp.empty()) // Le groupe est vide
    //continue;
    GroupPartInfo* gpi = new GroupPartInfo(grp, partid, m_use_degenerated_hexa);
    m_parts.add(gpi);
    ++partid;

    GroupPartInfo& current_grp = *gpi;
    // Il faut maintenant déterminer combien d'éléments de chaque type
    // ensight (tria3, hexa8, ...) on a dans le groupe.
    // two versions: if extra item types are added switch to an optimized version
    // (a large amount of types may be added, equal to the number of elements)
    if (ItemTypeMng::nbBuiltInItemType() == ItemTypeMng::nbBasicItemType()) // no extra type
    {
      debug(Trace::High) << "Using standard group part building algo";
      if (!grp.mesh()->itemTypeMng()->hasGeneralCells(grp.mesh())) {// classical types
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
          ENUMERATE_ITEM (iz, grp)
          {
            Item mi = *iz;
            if (mi.type() == type_to_seek) {
              ItemWithNodes e = mi.toItemWithNodes();
              items[index] = e;
              ++index;
            }
          }
        }
      }
      else {// polyhedral mesh items
        ENUMERATE_ITEM (i2, grp) {
          const Item& item = *i2;
          auto item_type = current_grp.generalItemTypeId(item);
          EnsightPart* ensight_part = current_grp.getTypeInfo(item_type);
          if (!ensight_part)
            continue ;
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
            continue ;
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
 * \brief Sauvegarde la connectivité des éléments d'un groupe.
 
 \relates DumpWEnsight7

 Sauve la connectivité des éléments du groupe \a ensight_grp. la difficulté
 vient du fait que Ensight nécessite qu'on sauve les éléments suivant leur
 type, c'est à dire les hexa d'un côté, les tétra de l'autre et ainsi
 de suite. \a ensight_grp.m_nb_sub_part[] contient pour le nombre d'éléments
 pour chaque types ensight. On parcours donc la liste des éléments du groupe
 autant de fois qu'il y a de types ensight possible (4 pour les mailles,
 2 pour les faces) et à chaque passe on ne sauve que les éléments qui sont
 du bon type. C'est un peu fastidieux mais cela évite d'avoir à gérer une
 liste pour chaque sous-type.

 \param ofile flot de sortie
 \param nodes_index indice de chaque noeud dans le tableau des coordonnées
 de Ensight. 
 \param ensight_grp  groupe à sauver
 \param wf écrivain
*/
void DumpWEnsight7::
_saveGroup(std::ostream& ofile, const GroupPartInfo& ensight_grp,
           ConstArrayView<Integer> nodes_index, WriteBase& wf)
{
  ItemGroup igrp = ensight_grp.group();

  writeFileString(ofile, "part");
  writeFileInt(ofile, ensight_grp.partId());
  if (isParallelOutput()) {
    // Dans le cas d'une sortie parallele, prefixe le nom du groupe par le
    // numéro du CPU.
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
    // Sauve les uniqueId() des entités
    for( ConstArrayView<Item>::const_iter i(items); i(); ++i ){
      Item mi = *i;
      writeFileInt(ofile,mi.uniqueId()+1);
    }
#endif

    if (type_name == "nfaced") {
      // Traitement particulier pour les mailles non standards
      // Comme nos faces ne sont pas orientées par rapport à une
      // maille, on utilise la connectivité locale de chaque face.
      // 1. Sauve le nombre de faces de chaque élément
      {
        if (!m_mesh->itemTypeMng()->hasGeneralCells(m_mesh)) {
          // Tous les éléments ont le même nombre de faces
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
      // 2. Sauve pour chaque elément, le nombre de noeuds de chacune
      //    de ces faces
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
      // 3. Sauve pour chaque face de chaque élément la liste de ces
      //    noeuds
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
              // Inversion du sens des noeuds
              // A priori, il y a un bug dans Ensight (7.4.1(g)) concernant les
              // intersections de ce type d'éléments avec les objets
              // Ensight (plan, sphere, ...). Quelle que soit l'orientation
              // des faces retenues, le comportant n'est pas correcte.
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
      // Traitement particulier pour les faces à plus de 4 noeuds:

      // 1. Sauve pour chaque elément, le nombre de ses noeuds
      //cerr << "** NSIDED ELEMENT\n";
      for (Item mi : items) {
        ItemWithNodes e = mi.toItemWithNodes();
        Integer nb_node = e.nbNode();
        writeFileInt(ofile, nb_node);
      }
      // 2. Sauve pour chaque elément la liste de ces noeuds
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
      // Cas général

      Integer nb_node = type_info.nbNode();
      array_id.resize(nb_node);

      // Sauvegarde les éléments.

      // Sauve la connectivité des éléments
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
 * \brief Applique un fonctor sur les éléments d'un groupe.
 
 \relates DumpWEnsight7

 Applique le fonctor \a from_func sur le groupe \a ensight_grp.

 Le fonctor doit dériver de WriteBase et est utilisé pour sauver
 une variable d'un type donné. Dans Ensight, les variables sont sauvées pour
 chaque partie et chaque type d'éléments (hexa8, ...). Le fonctionnement
 est similaire à celui de al sauvegarde de la géométrie (voir
 la fonction _ensightSaveGroup()). Il faut donc procéder de la même
 manière, à savoir itérer sur le groupe autant de fois qu'il y a de type
 d'élément et ne sauvegarde à une itération que les variables pour
 un élément de type donné.

 \warning Cette fonction ne supportera pas le traitement des variables
 définies sur un groupe autre que l'ensemble des éléments.

 \sa WriteFunctor

 \param ofile        flot de sortie
 \param ensight_grp  groupe
 \param from_func    fonctor
*/
void DumpWEnsight7::
_saveVariableOnGroup(std::ostream& ofile, const GroupPartInfo& ensight_grp,
                     WriteBase& from_func)
{
  ItemGroup igrp = ensight_grp.group();

  writeFileString(ofile, "part");
  writeFileInt(ofile, ensight_grp.partId());

  // Les éléments de même types (en particulier les 'nfaced', 'haxa8' ou 'nsided')
  // doivent être écrits d'un seul bloc (donc label de type unifié).
  // Par ailleurs, les données vectorielles doivent être entrelacés.
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
      // Cloture la précédente écriture
      if (func.get()) {
        func->end();
        func->putValue(ofile);
      }
      last_type_name = type_name;
      writeFileString(ofile, type_name);
      func = from_func.clone();
      func->begin();
    }

    // Sauvegarde les variables.
    func->write(items);
  }

  // Cloture finale
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
 * \brief Abstraction d'un fichier de sortie pour ensight.
 *
 * Dans le cas séquentiel, il s'agit réellement d'un fichier. Dans le cas
 * parallèle, il peut s'agir d'un fichier ou d'un flot mémoire suivant qu'on
 * est un processeur maître ou esclave.
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
      // TODO attention fuites mémoires.
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

        // Le maitre parcours tous les processeurs et demande pour chacun
        // d'eux la taille de chaque sauvegarde et si elle n'est pas nulle,
        // réceptionne le message.
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
        // Un esclave envoie au processeur maître (actuellement le processeur 0)
        // la taille de ses données puis ces données.
        // Il faut bien prendre la longueur de la chaîne retournée par m_strstream
        // et pas uniquement c_str() car le flux pouvant contenant des informations
        // au format binaire, on s'arrêterait au premier zéro.
        UniqueArray<int> len_array(1);
        std::string str = m_strstream.str();
        Integer len = arcaneCheckArraySize(str.length());
        UniqueArray<Byte> bytes(len);
        {
          // Recopie dans bytes la chaîne \a str
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

  // Détermine et créé le répertoire ou seront écrites les sorties
  // de cette itération.
  _buildPartDirectory();

  IMesh* mesh = m_mesh;

  // Récupère la liste des groupes et leur donne à chacun un identifiant
  // unique pour Ensight (part). Le numéro de l'identifiant débute à 2, le numéro 1
  // est pour la liste des noeuds
  {

    ItemGroupList list_group;
    // Si la liste des groupes à sauver est vide, sauve tous les groupes.
    // Sinon, sauve uniquement les groupes spécifiés.
    Integer nb_group = m_save_groups.count();
    if (nb_group == 0)
      list_group.clone(mesh->groups());
    else
      list_group.clone(m_save_groups);
    // Regarde la liste des variables partielles, et ajoute leur groupe à la liste
    // des groupes à sauver.
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

    // Récupère le nombre total d'eléments dans les groupes.
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

  // Sauvegarde la géométrie au format Ensight7 gold
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

    // Ce tableau sert pour chaque entité maille, face ou arête pour
    // référencer ses noeuds par rapport au tableau de coordonnées
    // utilisé par Ensight. Le premier élément de ce tableau à pour indice 1
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

      // Stocke les indices locaux pour Ensight
      {
        Integer ensight_index = 1;
        ENUMERATE_ITEM (i_item, all_nodes) {
          const Item& item = *i_item;
          all_nodes_index[item.localId()] = ensight_index;
          ++ensight_index;
        }
      }

      // Affiche les numéros unique des noeuds
#if 0
      {
        ENUMERATE_ITEM(i_item,all_nodes){
          const Item& item = *i_item;
          writeFileInt(wf.stream(),item.uniqueId()+1);
        }
      }
#endif
      // Affiche les coordonnées de chaque noeud
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
  // Sauvegarde des identifiants uniques
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

    // GG: NOTE: je ne suis pas sur que cela fonctionne bien.
    //TODO: supprimer l'utilisation de deux variables
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

  // Seul un processeur maître génère un fichier 'case'
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

    // Sauvegarde en temps. Pour paraview, il faut le mettre apres les variables
    m_case_file << "\nTIME\n";
    m_case_file << "time set:              1\n";
    m_case_file << "number of steps:       " << m_times.size() << '\n';
    m_case_file << "filename start number: 0\n";
    m_case_file << "filename increment:    1\n";
    m_case_file << "time values:\n";
    // Faire attention de ne pas dépasser les 79 caractères par ligne
    // autorisés par Ensight. Pour être sur, on n'écrit qu'un temps pas
    // ligne. Les temps sont sauvés le maximum de chiffres significatifs
    // car Ensight n'aime pas que deux temps soient égaux.
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
 * \brief Vérifie la validité de la variable à sauvegarder
 *
 * Seules les variables aux noeuds, aux faces ou aux mailles sont traitées.
 *
 * Seules les variables qui sont dans la liste #m_save_variable_list sont
 * sauvées. Si cette liste est vide, elle sont toutes sauvées.
 *
 * \retval true si la variable est valide
 * \retval false sinon
 */
bool DumpWEnsight7::
_isValidVariable(IVariable& v) const
{
  if (!v.isUsed())
    return false;
  eItemKind ik = v.itemKind();
  if (ik != IK_Cell && ik != IK_Node && ik != IK_Face)
    return false;
  // Si une liste de variable est spécifiée, vérifie que la variable est dans
  // cette liste. Si cette liste est vide, on sauve automatiquement toutes les
  // variables.
  Integer nb_var = m_save_variables.count();
  if (nb_var != 0) {
    return m_save_variables.contains(&v);
  }
  return true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Construit le nom de fichier pour un nom de variable ou de maillage.
 *
 Construit le nom de fichier dans lequel sera sauvegardé une variable ou
 un maillage de nom \a name. Le nom du fichier est retourné dans \a filename.
 
 Le numéro de la protection est insérée à la fin du fichier sous la forme d'un nombre
 formaté avec #m_max_prots_digit \a caractères. Par exemple, pour 'Pression'
 à l'itération 4 avec #m_max_prot_digit égal à 6, on obtiendra le
 fichier de nom 'Pression000004'.
 \todo changer la determination du nom de fichier et utiliser Directory.
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
      ostr().fill('0'); // Caractère de remplissage.
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
 * \brief Construit le répertoire où seront sauvées les variables.
 *
 Si on souhaite les sorties sur plusieurs fichiers, le répertoire du bloc courant
 a pour nom 'bloc******'. Par exemple, pour le 4ème bloc, il s'agit de
 'bloc000004'.

 le numéro du bloc est insérée sous la forme d'un nombre
 formaté avec #m_max_prots_digit \a caractères. 
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
    ostr().fill('0'); // Caractère de remplissage.
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
    // Si le nombre de temps est inférieur à la taille d'un fileset-set,
    // il n'y a pas besoin de mettre une '*'. On peut directement
    // mettre des zéros. Cela est aussi indispensable pour empêcher
    // paraview de planter quand il relit ce genre de fichiers.
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
 * \brief Sauvegarde des variables scalaires.
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
 * \brief Sauvegarde des variables tableau de dimension constante (par ligne) de scalaires
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
 * \brief Sauvegarde des variables tableau de dimension non constante (par ligne) scalaires.
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
 * \brief Sauvegarde des variables vectorielles.
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
 * \brief Sauvegarde des variables tableau de vecteurs
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
 * \brief Sauvegarde des variables tableau de vecteurs
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
 * \brief Post-traitement au format Ensight7.
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
 * \brief Post-traitement au format Ensight7.
 */
class Ensight7PostProcessorServiceV2
: public ArcaneEnsight7PostProcessorObject
{
 public:

  typedef ArcaneEnsight7PostProcessorObject BaseType;

 public:

  Ensight7PostProcessorServiceV2(const ServiceBuildInfo& sbi)
  : ArcaneEnsight7PostProcessorObject(sbi)
  , m_mesh(sbi.mesh())
  , m_writer(nullptr)
  {
  }

  void build() override
  {
    PostProcessorWriterBase::build();
  }
  IDataWriter* dataWriter() override { return m_writer; }
  void notifyBeginWrite() override;
  void notifyEndWrite() override;
  void close() override {}

  void setBaseDirectoryName(const String& dirname) override
  {
    PostProcessorWriterBase::setBaseDirectoryName(dirname);
  }
  const String& baseDirectoryName() override
  {
    return PostProcessorWriterBase::baseDirectoryName();
  }
  void setMesh(IMesh* mesh) override
  {
    m_mesh = mesh;
  }
  void setTimes(RealConstArrayView times) override
  {
    PostProcessorWriterBase::setTimes(times);
  }
  void setVariables(VariableCollection variables) override
  {
    PostProcessorWriterBase::setVariables(variables);
  }
  void setGroups(ItemGroupCollection groups) override
  {
    PostProcessorWriterBase::setGroups(groups);
  }

 private:

  IMesh* m_mesh;
  DumpW* m_writer;
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
