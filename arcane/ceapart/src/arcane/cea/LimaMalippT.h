// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* LimaMalipp.cc                                               (C) 2000-2019 */
/*                                                                           */
/* Lecture d'un fichier au format Lima MLI ou MLI2.                          */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/CheckedConvert.h"

#include "arcane/IMeshReader.h"
#include "arcane/ISubDomain.h"
#include "arcane/IIOMng.h"
#include "arcane/IParallelMng.h"
#include "arcane/IPrimaryMesh.h"
#include "arcane/Item.h"
#include "arcane/ItemTypeMng.h"
#include "arcane/ItemGroup.h"
#include "arcane/ArcaneException.h"
#include "arcane/Service.h"
#include "arcane/Timer.h"
#include "arcane/ServiceFactory.h"
#include "arcane/ServiceInfo.h"
#include "arcane/CaseOptionsMain.h"
#include "arcane/MeshUtils.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/VariableTypes.h"
#include "arcane/ServiceBuildInfo.h"
#include "arcane/XmlNodeList.h"
#include "arcane/IXmlDocumentHolder.h"
#include "arcane/IItemFamily.h"
#include "arcane/FactoryService.h"
#include "arcane/AbstractService.h"
#include "arcane/MathUtils.h"

#include "arcane/cea/LimaCutInfosReader.h"

#include <memory>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using std::string;

template<typename LimaMaliReader>
class LimaGroupReader;

namespace LimaUtils
{
void createGroup(IItemFamily* family, const String& name, Int32ArrayView local_ids);
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LimaVolume
{
 public:
  static Real Hexaedron8Volume(Real3 n[8])
  {
    Real v1 = math::matDet((n[6] - n[1]) + (n[7] - n[0]), n[6] - n[3], n[2] - n[0]);
    Real v2 = math::matDet(n[7] - n[0], (n[6] - n[3]) + (n[5] - n[0]), n[6] - n[4]);
    Real v3 = math::matDet(n[6] - n[1], n[5] - n[0], (n[6] - n[4]) + (n[2] - n[0]));

    Real res = (v1 + v2 + v3) / 12.0;

    return res;
  }

  static Real Pyramid5Volume(Real3 n[5])
  {
    return math::matDet(n[1] - n[0], n[3] - n[0], n[4] - n[0]) +
    math::matDet(n[3] - n[2], n[1] - n[2], n[4] - n[2]);
  }

  static Real Quad4Surface(Real3 n[4])
  {
    Real x1 = n[1].x - n[0].x;
    Real y1 = n[1].y - n[0].y;
    Real x2 = n[2].x - n[1].x;
    Real y2 = n[2].y - n[1].y;
    Real surface = x1 * y2 - y1 * x2;

    x1 = n[2].x - n[0].x;
    y1 = n[2].y - n[0].y;
    x2 = n[3].x - n[2].x;
    y2 = n[3].y - n[2].y;

    surface += x1 * y2 - y1 * x2;

    return surface;
  }

  static Real Triangle3Surface(Real3 n[3])
  {
    Real x1 = n[1].x - n[0].x;
    Real y1 = n[1].y - n[0].y;
    Real x2 = n[2].x - n[1].x;
    Real y2 = n[2].y - n[1].y;

    return x1 * y2 - y1 * x2;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur des fichiers de maillage via la bibliothèque LIMA pour
 * fichier '.mli' ou 'mli2'.
 *
 * Le paramêtre template \a LimaMaliReader vaut soit Lima::MaliPPReader pour
 * les fichiers 'mli', soit Lima::MaliPPReader2 pour 'mli2'.
 */
template<typename LimaMaliReader>
class LimaMalippMeshBase
: public TraceAccessor
{
 public:
  LimaMalippMeshBase(ITraceMng* trace_mng)
  : TraceAccessor(trace_mng) {}
  virtual ~LimaMalippMeshBase() = default;

 public:
 public:
  virtual bool readMeshPart(ITimerMng* timer_mng, LimaMaliReader* reader, IPrimaryMesh* mesh,
                            const String& filename, Real length_multiplier) = 0;

 protected:
  void _createGroupFromUniqueIds(IMesh* mesh, const String& name, eItemKind ik,
                                 Int64ConstArrayView unique_ids);
  void _createGroupFromHashTable(IMesh* mesh, const String& name, eItemKind ik,
                                 Int64ConstArrayView unique_ids,
                                 const HashTableMapT<Int64, Int32>& converter);
  void _setCoords(Real3ConstArrayView coords, HashTableMapT<Int64, Real3>& nodes_coords,
                  Int64 current_unique_id);
  void _createGroups(IMesh* mesh, eItemKind item_kind, LimaGroupReader<LimaMaliReader>* reader,
                     HashTableMapT<Int64, Int32>* converter);
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur des fichiers de maillage via la bibliothèque LIMA.
 *
 * Le paramètre 'template' permet de spécifier un wrapper pour lire les
 * maillages 2D ou 3D.
 */
template <typename ReaderWrapper>
class LimaMalippReaderT
: public LimaMalippMeshBase<typename ReaderWrapper::LimaMaliReaderType>
{
  using TraceAccessor::info;
  using TraceAccessor::log;
  using TraceAccessor::logdate;
 public:
  typedef typename ReaderWrapper::LimaMaliReaderType LimaMaliReader;
 public:
  LimaMalippReaderT(IParallelMng* pm)
  : LimaMalippMeshBase<LimaMaliReader>(pm->traceMng()),
    m_cut_infos_reader(new LimaCutInfosReader(pm))
  {}

  ~LimaMalippReaderT() override
  {
    delete m_cut_infos_reader;
  }

 public:
 public:
  bool readMeshPart(ITimerMng* timer_mng,LimaMaliReader* reader,
                    IPrimaryMesh* mesh, const String& filename,
                    Real length_multiplier) override;

 private:
  LimaCutInfosReader* m_cut_infos_reader;
  ReaderWrapper m_wrapper;

  bool _readMeshPart(ITimerMng* timer_mng,LimaMaliReader* reader, IPrimaryMesh* mesh,
                     const String& filename, Real length_multiplier);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LimaMaliReader>
class LimaMalippReaderWrapper
{
 public:
  typedef LimaMaliReader LimaMaliReaderType;
  typedef typename LimaMaliReader::NuageReader LimaNodeGroup;

 public:
  LimaMalippReaderWrapper()
  : m_mali_reader(0)
  {}
  void setReader(LimaMaliReader* reader)
  {
    m_mali_reader = reader;
  }
  LimaMaliReader* reader()
  {
    return m_mali_reader;
  }

 public:
  Lima::size_type
  _readGroup(typename LimaMaliReader::SurfaceReader reader,
             Lima::size_type begin, Lima::size_type n, Lima::size_type* buffer)
  {
    return reader.lire_mailles_ids(begin, n, buffer);
  }
  Lima::size_type
  _readGroup(typename LimaMaliReader::VolumeReader reader,
             Lima::size_type begin, Lima::size_type n, Lima::size_type* buffer)
  {
    return reader.lire_mailles_ids(begin, n, buffer);
  }
  Lima::size_type
  _readGroup(typename LimaMaliReader::LigneReader reader,
             Lima::size_type begin, Lima::size_type n, Lima::size_type* buffer)
  {
    return reader.lire_bras_ids(begin, n, buffer);
  }

  Lima::size_type
  _readGroup(typename LimaMaliReader::NuageReader reader,
             Lima::size_type begin, Lima::size_type n, Lima::size_type* buffer)
  {
    return reader.lire_noeuds_ids(begin, n, buffer);
  }

  double* allocateNodesCoordsBuffer(Lima::size_type buf_size)
  {
    return m_mali_reader->allouer_tampon_coords(buf_size);
  }
  Lima::size_type* allocateNodesIdBuffer(Lima::size_type buf_size)
  {
    return m_mali_reader->allouer_tampon_ids(buf_size);
  }
  Lima::size_type readNodes(Lima::size_type begin, Lima::size_type count,
                            Lima::size_type* ids, double* coords)
  {
    return m_mali_reader->lire_noeuds(begin, count, ids, coords);
  }

  template <typename LimaGroupReader>
  void readGroup(LimaGroupReader& reader, Int64Array& items_unique_id)
  {
    using namespace Lima;
    //cout << "----------------------------------------------------------"
    //     << reader.nom();
    size_type* buffer = 0;
    const Integer nb_item = CheckedConvert::toInteger(reader.composition().nb_elements);
    items_unique_id.resize(nb_item);
    const Integer step_size = nb_item > 10000 ? nb_item / 10 : nb_item;
    buffer = reader.allouer_tampon_ids(step_size);
    Integer begin = 0;
    for (Integer i = 0; i * step_size < nb_item; ++i) {
      const Integer count = CheckedConvert::toInteger(_readGroup(reader, (size_type)(i * step_size), (size_type)step_size, buffer));
      size_type* ptr = buffer;
      for (Integer n = 0; n < count; ++n) {
        size_type lima_id = *ptr++;
        items_unique_id[begin + n] = lima_id - 1;
        //cout << lima_id << " ";
      }
      begin += count;
    }
    delete[] buffer;
    //cout << endl;
  }

 protected:
  LimaMaliReader* m_mali_reader;
  ScopedPtrT<LimaGroupReader<LimaMaliReader>> m_cell_group_reader;
  ScopedPtrT<LimaGroupReader<LimaMaliReader>> m_node_group_reader;
  ScopedPtrT<LimaGroupReader<LimaMaliReader>> m_face_group_reader;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LimaMaliReader>
class LimaGroupReader
{
 public:
  LimaGroupReader(LimaMalippReaderWrapper<LimaMaliReader>* wrapper)
  : m_wrapper(wrapper)
  {}
  virtual ~LimaGroupReader() {}
  virtual void read(const String& name, Int64Array& items_unique_id) = 0;

 public:
  virtual StringConstArrayView groupsName()
  {
    std::vector<std::string> groups;
    _getGroupsName(groups);
    m_groups_name.clear();
    for (size_t i = 0, is = groups.size(); i < is; ++i)
      m_groups_name.add(String(groups[i]));
    return m_groups_name;
  }

 protected:
  virtual void _getGroupsName(std::vector<std::string>& groups) = 0;

 protected:
  LimaMalippReaderWrapper<LimaMaliReader>* m_wrapper;

 private:
  StringUniqueArray m_groups_name;
};

template<typename LimaMaliReader>
class NodeGroupReader
: public LimaGroupReader<LimaMaliReader>
{
 public:
  NodeGroupReader(LimaMalippReaderWrapper<LimaMaliReader>* wrapper)
  : LimaGroupReader<LimaMaliReader>(wrapper)
  {}
  virtual void read(const String& name, Int64Array& items_unique_id)
  {
    typename LimaMaliReader::NuageReader r = this->m_wrapper->reader()->nuage(name.localstr());
    this->m_wrapper->readGroup(r, items_unique_id);
  }
  virtual void _getGroupsName(std::vector<std::string>& groups)
  {
    this->m_wrapper->reader()->liste_nuages(groups);
  }

 private:
};

template<typename LimaMaliReader>
class EdgeGroupReader
: public LimaGroupReader<LimaMaliReader>
{
 public:
  EdgeGroupReader(LimaMalippReaderWrapper<LimaMaliReader>* wrapper)
  : LimaGroupReader<LimaMaliReader>(wrapper)
  {}
  virtual void read(const String& name, Int64Array& items_unique_id)
  {
    typename LimaMaliReader::LigneReader r = this->m_wrapper->reader()->ligne(name.localstr());
    this->m_wrapper->readGroup(r, items_unique_id);
  }
  virtual void _getGroupsName(std::vector<std::string>& groups)
  {
    this->m_wrapper->reader()->liste_lignes(groups);
  }

 private:
};

template<typename LimaMaliReader>
class FaceGroupReader
: public LimaGroupReader<LimaMaliReader>
{
 public:
  FaceGroupReader(LimaMalippReaderWrapper<LimaMaliReader>* wrapper)
  : LimaGroupReader<LimaMaliReader>(wrapper)
  {
  }
  virtual void read(const String& name, Int64Array& items_unique_id)
  {
    typename LimaMaliReader::SurfaceReader r = this->m_wrapper->reader()->surface(name.localstr());
    this->m_wrapper->readGroup(r, items_unique_id);
  }
  virtual void _getGroupsName(std::vector<std::string>& groups)
  {
    this->m_wrapper->reader()->liste_surfaces(groups);
  }
};

template<typename LimaMaliReader>
class CellGroupReader
: public LimaGroupReader<LimaMaliReader>
{
 public:
  CellGroupReader(LimaMalippReaderWrapper<LimaMaliReader>* wrapper)
  : LimaGroupReader<LimaMaliReader>(wrapper)
  {
  }
  virtual void read(const String& name, Int64Array& items_unique_id)
  {
    typename LimaMaliReader::VolumeReader r = this->m_wrapper->reader()->volume(name.localstr());
    this->m_wrapper->readGroup(r, items_unique_id);
  }
  virtual void _getGroupsName(std::vector<std::string>& groups)
  {
    this->m_wrapper->reader()->liste_volumes(groups);
  }

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LimaMaliReader>
class LimaMalipp2DReaderWrapper
: public LimaMalippReaderWrapper<LimaMaliReader>
{
  typedef LimaMalippReaderWrapper<LimaMaliReader> BaseClass;
  using BaseClass::m_mali_reader;
  using BaseClass::m_cell_group_reader;
  using BaseClass::m_node_group_reader;
  using BaseClass::m_face_group_reader;
 public:
  typedef Lima::Composition LimaComposition;
  typedef typename LimaMaliReader::SurfaceReader LimaCellGroup;
  typedef typename LimaMaliReader::LigneReader LimaFaceGroup;
  typedef typename LimaMaliReader::NuageReader LimaNodeGroup;
 public:
  LimaComposition cells()
  {
    return m_mali_reader->composition_polygones();
  }

  LimaComposition faces()
  {
    return m_mali_reader->composition_bras();
  }

  LimaComposition nodes()
  {
    return m_mali_reader->composition_noeuds();
  }

  std::vector<std::string> cellGroups()
  {
    std::vector<std::string> groups;
    m_mali_reader->liste_surfaces(groups);
    return groups;
  }

  LimaGroupReader<LimaMaliReader>* cellGroupReader()
  {
    if (!m_cell_group_reader.get())
      m_cell_group_reader = new FaceGroupReader<LimaMaliReader>(this);
    return m_cell_group_reader.get();
  }

  LimaGroupReader<LimaMaliReader>* nodeGroupReader()
  {
    if (!m_node_group_reader.get())
      m_node_group_reader = new NodeGroupReader<LimaMaliReader>(this);
    return m_node_group_reader.get();
  }

  LimaGroupReader<LimaMaliReader>* faceGroupReader()
  {
    if (!m_face_group_reader.get())
      m_face_group_reader = new EdgeGroupReader<LimaMaliReader>(this);
    return m_face_group_reader.get();
  }

  std::vector<std::string> faceGroups()
  {
    std::vector<std::string> groups;
    m_mali_reader->liste_lignes(groups);
    return groups;
  }

  std::vector<std::string> nodeGroups()
  {
    std::vector<std::string> groups;
    m_mali_reader->liste_nuages(groups);
    return groups;
  }

  LimaCellGroup cellGroup(const string& name)
  {
    return m_mali_reader->surface(name);
  }

  LimaFaceGroup faceGroup(const string& name)
  {
    return m_mali_reader->ligne(name);
  }

  LimaNodeGroup nodeGroup(const string& name)
  {
    return m_mali_reader->nuage(name);
  }

  Lima::size_type* allocateCellsBuffer(Lima::size_type buf_size)
  {
    return m_mali_reader->allouer_tampon_polygones(buf_size);
  }
  Lima::size_type readCells(Lima::size_type begin, Lima::size_type count, Lima::size_type* buffer)
  {
    return m_mali_reader->lire_polygones(begin, count, buffer);
  }
  Integer facesBufferSize(Lima::size_type buf_size)
  {
    // Normalement il faut utiliser m_mali_reader->allouer_tampon_bras
    // mais seul le proc maitre peut le faire et les autres ont besoin
    // aussi d'allouer un tampon.
    //return m_mali_reader->allouer_tampon_bras(buf_size);
    return CheckedConvert::toInteger(8 * buf_size);
  }
  Lima::size_type readFaces(Lima::size_type begin, Lima::size_type count, Lima::size_type* buffer)
  {
    return m_mali_reader->lire_bras(begin, count, buffer);
  }

  int limaDimension()
  {
    return Lima::D2;
  }
  Integer dimension()
  {
    return 2;
  }
  const char* strDimension()
  {
    return "2D";
  }
  Real3 readNodeCoords(const double* ptr)
  {
    return Real3(ptr[0], ptr[1], 0.0);
  }
  static Integer cellToType(Integer nb_node)
  {
    switch (nb_node) {
    case 3:
      return IT_Triangle3;
    case 4:
      return IT_Quad4;
    case 5:
      return IT_Pentagon5;
    case 6:
      return IT_Hexagon6;
    default:
      break;
    }
    return IT_NullType;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LimaMaliReader>
class LimaMalipp3DReaderWrapper
: public LimaMalippReaderWrapper<LimaMaliReader>
{
  typedef LimaMalippReaderWrapper<LimaMaliReader> BaseClass;
  using BaseClass::m_mali_reader;
  using BaseClass::m_cell_group_reader;
  using BaseClass::m_node_group_reader;
  using BaseClass::m_face_group_reader;
 public:
  typedef typename LimaMaliReader::VolumeReader LimaCellGroup;
  typedef typename LimaMaliReader::SurfaceReader LimaFaceGroup;
  typedef typename LimaMaliReader::NuageReader LimaNodeGroup;
  typedef Lima::Composition LimaComposition;

 public:
  LimaComposition cells()
  {
    return m_mali_reader->composition_polyedres();
  }
  LimaComposition faces()
  {
    return m_mali_reader->composition_polygones();
  }
  LimaComposition nodes()
  {
    return m_mali_reader->composition_noeuds();
  }

 public:
 public:
  //typedef Lima::Volume LimaCellGroup;
  typedef Lima::Polyedre LimaCell;
  //typedef Lima::Surface LimaFaceGroup;
  typedef Lima::Polygone LimaFace;

  LimaGroupReader<LimaMaliReader>* cellGroupReader()
  {
    if (!m_cell_group_reader.get())
      m_cell_group_reader = new CellGroupReader<LimaMaliReader>(this);
    return m_cell_group_reader.get();
  }

  LimaGroupReader<LimaMaliReader>* faceGroupReader()
  {
    if (!m_face_group_reader.get())
      m_face_group_reader = new FaceGroupReader<LimaMaliReader>(this);
    return m_face_group_reader.get();
  }

  LimaGroupReader<LimaMaliReader>* nodeGroupReader()
  {
    if (!m_node_group_reader.get())
      m_node_group_reader = new NodeGroupReader<LimaMaliReader>(this);
    return m_node_group_reader.get();
  }

  std::vector<std::string> cellGroups()
  {
    std::vector<std::string> groups;
    m_mali_reader->liste_volumes(groups);
    return groups;
  }

  std::vector<std::string> faceGroups()
  {
    std::vector<std::string> groups;
    m_mali_reader->liste_surfaces(groups);
    return groups;
  }

  std::vector<std::string> nodeGroups()
  {
    std::vector<std::string> groups;
    m_mali_reader->liste_nuages(groups);
    return groups;
  }

  LimaCellGroup cellGroup(const string& name)
  {
    return m_mali_reader->volume(name);
  }
  LimaFaceGroup faceGroup(const string& name)
  {
    return m_mali_reader->surface(name);
  }
  LimaNodeGroup nodeGroup(const string& name)
  {
    return m_mali_reader->nuage(name);
  }
  Lima::size_type* allocateCellsBuffer(Lima::size_type buf_size)
  {
    return m_mali_reader->allouer_tampon_polyedres(buf_size);
  }
  Lima::size_type readCells(Lima::size_type begin, Lima::size_type count, Lima::size_type* buffer)
  {
    return m_mali_reader->lire_polyedres(begin, count, buffer);
  }
  Integer facesBufferSize(Lima::size_type buf_size)
  {
    // Normalement il faut utiliser m_mali_reader->allouer_tampon_polygone
    // mais seul le proc maitre peut le faire et les autres ont besoin
    // aussi d'allouer un tampon.
    //return m_mali_reader->allouer_tampon_polygones(buf_size);
    return CheckedConvert::toInteger(14 * buf_size);
  }
  Lima::size_type readFaces(Lima::size_type begin, Lima::size_type count, Lima::size_type* buffer)
  {
    return m_mali_reader->lire_polygones(begin, count, buffer);
  }

  int limaDimension()
  {
    return Lima::D3;
  }
  Integer dimension()
  {
    return 3;
  }
  const char* strDimension()
  {
    return "3D";
  }
  Real3 readNodeCoords(const double* ptr)
  {
    return Real3(ptr[0], ptr[1], ptr[2]);
  }
  static Integer cellToType(Integer nb_node)
  {
    switch (nb_node) {
    case 4:
      return IT_Tetraedron4;
    case 5:
      return IT_Pyramid5;
    case 6:
      return IT_Pentaedron6;
    case 8:
      return IT_Hexaedron8;
    case 10:
      return IT_Heptaedron10;
    case 12:
      return IT_Octaedron12;
    default:
      break;
    }
    return IT_NullType;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur des fichiers de maillage via la bibliothèque LIMA.
 */
template<typename LimaMaliReader>
class LimaMalippReader
: public TraceAccessor
{
 public:
  LimaMalippReader(ITraceMng* trace_mng)
  : TraceAccessor(trace_mng) {}

 public:
  IMeshReader::eReturnType
  readMeshFromFile(ITimerMng* tm,IPrimaryMesh* mesh,
                   const String& file_name, Real length_multiplier);
 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename LimaMaliReader>
IMeshReader::eReturnType LimaMalippReader<LimaMaliReader>::
readMeshFromFile(ITimerMng* timer_mng,IPrimaryMesh* mesh,
                 const String& filename,Real length_multiplier)
{
  if (filename.null() || filename.empty())
    return IMeshReader::RTIrrelevant;

  IParallelMng* pm = mesh->parallelMng();
  ScopedPtrT<LimaMaliReader> reader;
  bool is_master_io = pm->isMasterIO();
  Int32 master_io_rank = pm->masterIORank();
  Integer dimension = 0;
  if (is_master_io) {
    try {
      reader = new LimaMaliReader(filename.localstr(), 1);
    }
    catch (const Lima::erreur& ex) {
      ARCANE_FATAL("Impossible de lire le fichier MLI Lima <{0}> :",filename,ex.what());
    }
    catch (...) {
      ARCANE_FATAL("Impossible de lire le fichier MLI Lima <{0}>",filename);
    }
    dimension = reader->dimension();
    pm->broadcast(IntegerArrayView(1, &dimension), master_io_rank);
  }
  else
    pm->broadcast(IntegerArrayView(1, &dimension), master_io_rank);
  ScopedPtrT<LimaMalippMeshBase<LimaMaliReader>> lm;
  if (dimension == Lima::D3) {
    info() << "Maillage 3D";
    mesh->setDimension(3);
    lm = new LimaMalippReaderT<LimaMalipp3DReaderWrapper<LimaMaliReader>>(pm);
  }
  else if (dimension == Lima::D2) {
    info() << "Maillage 2D";
    mesh->setDimension(2);
    lm = new LimaMalippReaderT<LimaMalipp2DReaderWrapper<LimaMaliReader>>(pm);
  }
  else
    throw NotSupportedException(A_FUNCINFO, "Can not read Lima 1D mesh");

  if (!lm.get()) {
    log() << "Dimension du maillage non reconnue par lima";
    return IMeshReader::RTIrrelevant;
  }
  bool ret = lm->readMeshPart(timer_mng,reader.get(), mesh, filename, length_multiplier);
  if (ret)
    return IMeshReader::RTError;
  return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ReaderWrapper>
bool LimaMalippReaderT<ReaderWrapper>::
readMeshPart(ITimerMng* timer_mng,LimaMaliReader* reader, IPrimaryMesh* mesh,
             const String& filename, Real length_multiplier)
{
  return _readMeshPart(timer_mng, reader, mesh, filename, length_multiplier);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecture du maillage.
 * Seul le proc maitre a une instance de \a reader non nulle. Les autres
 * ne doivent pas l'utiliser.
 */
template <typename ReaderWrapper>
bool LimaMalippReaderT<ReaderWrapper>::
_readMeshPart(ITimerMng* timer_mng, LimaMaliReader* reader, IPrimaryMesh* mesh,
              const String& file_name,Real length_multiplier)
{
  typedef Lima::size_type size_type;
  size_type basic_step = 100000;

  IParallelMng* pm = mesh->parallelMng();
  bool is_master_io = pm->isMasterIO();
  Integer master_rank = pm->masterIORank();
  Int32 nb_rank = pm->commSize();

  Integer nb_edge = 0;

  this->pwarning() << "Chargement Lima du fichier USING MALIPP avec partitionnement '" << file_name << '"';
  if (basic_step < 100000)
    this->pwarning() << "Small basic_step value=" << basic_step;

  const char* version = Lima::lima_version();
  info() << "Utilisation de la version " << version << " de Lima";

  Timer time_to_read(timer_mng, "ReadLima", Timer::TimerReal);

  this->log() << "Début lecture fichier " << file_name;

  ReaderWrapper wrapper;
  wrapper.setReader(reader);

  if (reader && reader->dimension() != wrapper.limaDimension())
    ARCANE_FATAL("Le fichier n'est pas un maillage {0}",wrapper.strDimension());

  bool is_3d = (mesh->dimension() == 3);

  Int64 mesh_nb_node = 0;
  Int64 mesh_nb_cell = 0;
  Int64 mesh_nb_face = 0;

  // Le proc maitre lit le nombre d'entites et l'envoie aux autres
  if (is_master_io) {
    Lima::Composition lima_cells = wrapper.cells();
    Lima::Composition lima_nodes = wrapper.nodes();
    Lima::Composition lima_faces = wrapper.faces();

    mesh_nb_node = (Int64)lima_nodes.nb_elements;
    mesh_nb_cell = (Int64)lima_cells.nb_elements;
    mesh_nb_face = (Int64)lima_faces.nb_elements;
    Int64 nb_items[3];
    nb_items[0] = mesh_nb_node;
    nb_items[1] = mesh_nb_cell;
    nb_items[2] = mesh_nb_face;
    pm->broadcast(Int64ArrayView(3, nb_items), master_rank);

    info() << "Unité de longueur du fichier: " << reader->unite_longueur();
    if (length_multiplier == 0.0)
      length_multiplier = 1.0;
    else
      length_multiplier *= reader->unite_longueur();
    pm->broadcast(RealArrayView(1, &length_multiplier), master_rank);
  }
  else {
    Int64 nb_items[3];
    pm->broadcast(Int64ArrayView(3, nb_items), master_rank);
    mesh_nb_node = nb_items[0];
    mesh_nb_cell = nb_items[1];
    mesh_nb_face = nb_items[2];
    pm->broadcast(RealArrayView(1, &length_multiplier), master_rank);
  }

  size_type lima_nb_node = (size_type)mesh_nb_node;
  nb_edge = 0; //lima.nb_bras();

  info() << "-- Informations sur le maillage (Lima):";
  info() << "Nombre de noeuds  " << mesh_nb_node;
  info() << "Nombre d'arêtes   " << nb_edge;
  info() << "Nombre de faces   " << mesh_nb_face;
  info() << "Nombre de mailles " << mesh_nb_cell;
  if (mesh_nb_node == 0) {
    ARCANE_FATAL("Pas de noeuds dans le fichier de maillage.");
  }

  // Pour l'instant, laisse à false.
  // Si true, les uid sont incrémentés pour commencer à un.
  Int64 uid_to_add = 0;
  if (!platform::getEnvironmentVariable("ARCANE_LIMA_UNIQUE_ID").null()) {
    uid_to_add = 1;
    info() << "WARNING: UniqueId begin at 1";
  }

  Integer average_nb_cell = CheckedConvert::toInteger(mesh_nb_cell / nb_rank);

  UniqueArray<Int64> own_cells_infos;
  Integer own_nb_cell = 0;

  // Remplit le tableau contenant les infos des mailles
  if (is_master_io) {
    // Tableau contenant les infos aux mailles (voir IMesh::allocateMesh())
    Int64UniqueArray cells_infos;
    cells_infos.reserve(average_nb_cell * 2);

    Integer cell_step = average_nb_cell;
    if (cell_step == 0)
      ARCANE_FATAL("Number of cells is less than number of sub domains");

    //size_type step_size = (lima_nb_cell>basic_step) ? basic_step : mesh_nb_cell;
    size_type* buffer = wrapper.allocateCellsBuffer(cell_step + nb_rank);
    Int64 total_nb_cell = mesh_nb_cell;
    Integer current_nb_cell = 0;
    Integer total_count = 0;
    Integer last_print = 0;
    for (Integer i = 0; total_count < total_nb_cell; ++i) {
      if (i >= nb_rank)
        ARCANE_FATAL("Too many count reading cells i={0} nrank=",i,nb_rank);
      Int64 wanted_count = cell_step;
      // Le dernier sous-domaine prend toutes les mailles restantes
      if ((i + 1) == nb_rank)
        wanted_count = mesh_nb_cell;
      Integer count = CheckedConvert::toInteger(wrapper.readCells(i * cell_step, wanted_count, buffer));
      // Afficher un pourcentage de progression (tous les 5%)
      Integer p = (i * 20) / nb_rank;
      if (p > last_print) {
        last_print = p;
        info() << "Reading cells rank=" << i << " n=" << wanted_count << " count=" << count
               << " (" << (p * 5) << "%)";
      }
      current_nb_cell = count;
      total_count += count;
      size_type* ptr = buffer;
      cells_infos.clear();
      for (Integer p = 0; p < count; ++p) {
        Integer cell_local_id = CheckedConvert::toInteger(*ptr++);
        const Lima::size_type nodeCount = *ptr++;
        Integer n = CheckedConvert::toInteger(nodeCount);
        Integer ct = ReaderWrapper::cellToType(n);
        if (ct == IT_NullType)
          throw UnknownItemTypeException("LimaMaliPP::readFile: Cell", n, cell_local_id);
        // Stocke le type de la maille
        cells_infos.add(ct);
        // Stocke le numéro unique de la maille
        cells_infos.add(uid_to_add + cell_local_id - 1);

        for (Lima::size_type z = 0; z < nodeCount; ++z) {
          Int64 node_local_id = *ptr++;
          cells_infos.add(uid_to_add + node_local_id - 1);
        }
      }
      if (i != master_rank) {
        Integer nb_cells_infos[2];
        nb_cells_infos[0] = current_nb_cell;
        nb_cells_infos[1] = cells_infos.size();
        pm->send(IntegerConstArrayView(2, nb_cells_infos), i);
        pm->send(cells_infos, i);
      }
      else {
        own_nb_cell = current_nb_cell;
        own_cells_infos = cells_infos;
      }
    } // for (i = 0; i * stepSize < polygonNumber; i++)
    delete[] buffer;
  }
  else {
    Integer nb_cells_infos[2];
    pm->recv(IntegerArrayView(2, nb_cells_infos), master_rank);
    own_nb_cell = nb_cells_infos[0];
    own_cells_infos.resize(nb_cells_infos[1]);
    pm->recv(own_cells_infos, master_rank);
  }
  ItemTypeMng* itm = mesh->itemTypeMng();

  info() << " READ COORDINATES 1";

  HashTableMapT<Int64, Real3> nodes_coords(own_nb_cell * 10, true);
  // Remplit la table de hachage avec la liste des noeuds du sous-domaine
  {
    Integer cell_index = 0;
    for (Integer i = 0; i < own_nb_cell; ++i) {
      Integer type_id = (Integer)own_cells_infos[cell_index];
      ++cell_index;
      //Int64 cell_uid = own_cells_infos[cell_index];
      ++cell_index;
      ItemTypeInfo* it = itm->typeFromId(type_id);
      Integer current_nb_node = it->nbLocalNode();
      for (Integer z = 0; z < current_nb_node; ++z) {
        Int64 node_uid = own_cells_infos[cell_index + z];
        nodes_coords.add(node_uid, Real3());
      }
      cell_index += current_nb_node;
    }
  }

  info() << " READ COORDINATES 2";

  // Lecture des coordonnées
  if (is_master_io) {
    size_type step_size = (lima_nb_node > basic_step) ? basic_step : mesh_nb_node;
    size_type nodeCount = 0;
    size_type* idBuffer = reader->allouer_tampon_ids(step_size);
    double* coordsBuffer = reader->allouer_tampon_coords(step_size);
    Integer dim_step = reader->dimension();
    Real3UniqueArray current_coords;
    for (Integer i = 0; i < mesh_nb_node;) {
      Integer count = CheckedConvert::toInteger(reader->lire_noeuds(i, step_size, idBuffer, coordsBuffer));
      current_coords.clear();
      for (Integer n = 0; n < count; ++n, ++nodeCount) {
        Real3 coord;
        //Integer local_id = idBuffer[n] - 1;
        switch (dim_step) {
        case 1:
          coord.x = coordsBuffer[n];
          break;
        case 2:
          coord.x = coordsBuffer[2 * n];
          coord.y = coordsBuffer[(2 * n) + 1];
          break;
        case 3:
          coord.x = coordsBuffer[3 * n];
          coord.y = coordsBuffer[(3 * n) + 1];
          coord.z = coordsBuffer[(3 * n) + 2];
          break;
        }
        if (length_multiplier != 1.0)
          current_coords.add(coord * length_multiplier);
        else
          current_coords.add(coord);
      } // for (size_type n = 0; n < count; n++, nodeCount++)
      Integer sizes_info[1];
      sizes_info[0] = count;
      pm->broadcast(IntegerArrayView(1, sizes_info), master_rank);
      //warning() << "SEND COUNT=" << count << '\n';
      pm->broadcast(current_coords, master_rank);
      this->_setCoords(current_coords, nodes_coords, i + uid_to_add);
      i += count;
    } // for (i = 0; i < nodes.nb_elements; )
    delete[] idBuffer;
    delete[] coordsBuffer;
  }
  else {
    Real3UniqueArray current_coords;
    for (Int64 i = 0; i < mesh_nb_node;) {
      Integer sizes_info[1];
      pm->broadcast(IntegerArrayView(1, sizes_info), master_rank);
      Integer count = sizes_info[0];
      current_coords.resize(count);
      //warning() << "RESIZE COUNT=" << count << " begin=" << current_coords.begin() << '\n';
      pm->broadcast(current_coords, master_rank);
      this->_setCoords(current_coords, nodes_coords, i + uid_to_add);
      i += count;
    }
  }

  info() << " READ CELLS";

  // On a la liste des mailles et les coordonnées des noeuds associés.
  // Vérifie que le volume de la maille est positif, sinon inverse
  // la topologie de la maille.
  // Pour l'instant, traite uniquement les IT_Hexaedron8 et les pyramides
  {
    Integer cell_index = 0;
    Real3UniqueArray local_coords;
    local_coords.resize(8);
    Integer nb_reoriented = 0;
    for (Integer i = 0; i < own_nb_cell; ++i) {
      Integer type_id = (Integer)own_cells_infos[cell_index];
      ++cell_index;
      Int64 cell_uid = own_cells_infos[cell_index];
      ++cell_index;
      ItemTypeInfo* it = itm->typeFromId(type_id);
      Integer current_nb_node = it->nbLocalNode();
      local_coords.resize(current_nb_node);
      for (Integer z = 0; z < current_nb_node; ++z)
        local_coords[z] = nodes_coords.lookupValue(own_cells_infos[cell_index + z]);
      if (type_id == IT_Hexaedron8) {
        Real volume = LimaVolume::Hexaedron8Volume(local_coords.data());
        if (volume < 0.0) {
          std::swap(own_cells_infos[cell_index + 0], own_cells_infos[cell_index + 1]);
          std::swap(own_cells_infos[cell_index + 3], own_cells_infos[cell_index + 2]);
          std::swap(own_cells_infos[cell_index + 4], own_cells_infos[cell_index + 5]);
          std::swap(own_cells_infos[cell_index + 7], own_cells_infos[cell_index + 6]);
          info() << "Volume negatif Hexaedron8 uid=" << cell_uid << " v=" << volume;
          ++nb_reoriented;
        }
      }
      else if (type_id == IT_Pyramid5) {
        Real volume = LimaVolume::Pyramid5Volume(local_coords.data());
        if (volume < 0.0) {
          std::swap(own_cells_infos[cell_index + 0], own_cells_infos[cell_index + 1]);
          std::swap(own_cells_infos[cell_index + 2], own_cells_infos[cell_index + 3]);
          info() << "Volume negatif Pyramid5 uid=" << cell_uid << " v=" << volume;
          ++nb_reoriented;
        }
      }
      else if (type_id == IT_Quad4) {
        Real surface = LimaVolume::Quad4Surface(local_coords.data());
        if (surface < 0.0) {
          std::swap(own_cells_infos[cell_index + 0], own_cells_infos[cell_index + 1]);
          std::swap(own_cells_infos[cell_index + 2], own_cells_infos[cell_index + 3]);
          info() << "Surface negative Quad4 uid=" << cell_uid << " v=" << surface;
          ++nb_reoriented;
        }
      }
      else if (type_id == IT_Triangle3) {
        Real surface = LimaVolume::Triangle3Surface(local_coords.data());
        if (surface < 0.0) {
          std::swap(own_cells_infos[cell_index + 0], own_cells_infos[cell_index + 1]);
          info() << "Surface negative Triangle3 uid=" << cell_uid << " v=" << surface;
          ++nb_reoriented;
        }
      }

      cell_index += current_nb_node;
    }
    info() << "NB reoriented cell = " << nb_reoriented;
  }

  logdate() << "Début allocation du maillage";
  mesh->allocateCells(own_nb_cell, own_cells_infos, false);
  logdate() << "Fin allocation du maillage";

  Int32 sid = pm->commRank();
  ItemInternalList cells(mesh->itemsInternal(IK_Cell));
  for (Integer i = 0, is = cells.size(); i < is; ++i)
    cells[i]->setOwner(sid, sid);

  mesh->endAllocate();

  // Après allocation, la couche de mailles fantômes est créée.
  // Cependant, seules les mailles ont le propriétaire correcte.
  // Pour les autres entités, il faut le mettre à jour.
  mesh->setOwnersFromCells();

  this->_createGroups(mesh, IK_Cell, wrapper.cellGroupReader(), 0);
  this->_createGroups(mesh, IK_Node, wrapper.nodeGroupReader(), 0);

  HashTableMapT<Int64, Int32> faces_local_id(own_nb_cell * 4, true); // Numéro de la face lima dans le maillage \a mesh

  {
    Int64UniqueArray faces_first_node_unique_id;
    IntegerUniqueArray faces_nb_node;

    Int64UniqueArray faces_nodes_unique_id;
    Int64UniqueArray orig_nodes_id;
    orig_nodes_id.reserve(100);
    IntegerUniqueArray face_nodes_index;
    face_nodes_index.reserve(100);

    ItemInternalList mesh_nodes(mesh->itemsInternal(IK_Node));

    size_type face_basic_step = basic_step;
    size_type step_size = (mesh_nb_face > (Int64)face_basic_step) ? face_basic_step : mesh_nb_face;
    UniqueArray<size_type> buffer(wrapper.facesBufferSize(step_size));
    size_type total_nb_face = mesh_nb_face;
    Int64 current_nb_face = 0;
    Integer nb_lima_face_in_sub_domain = 0;
    info() << "Total_nb_face2=" << total_nb_face;
    //TODO possible a paralleliser la lecture:
    //1. Le proc maitre lit les tableaux et les envoie a N proc (scatter)
    //2. Chaque proc se charge de reordonner les noeuds des faces qu'il a recu puis les renvoient a tout le monde (gather)
    for (size_type i = 0; (i * step_size) < total_nb_face; ++i) {
      Integer count = 0;
      if (is_master_io) {
        count = CheckedConvert::toInteger(wrapper.readFaces(i * step_size, step_size, buffer.data()));
        pm->broadcast(Int32ArrayView(1, &count), master_rank);
        pm->broadcast(buffer, master_rank);
      }
      else {
        pm->broadcast(Int32ArrayView(1, &count), master_rank);
        pm->broadcast(buffer, master_rank);
      }
      size_type* ptr = buffer.data();
      info() << " Read Face2 N=" << i << " count=" << count << " step_size=" << step_size;
      orig_nodes_id.clear();
      faces_first_node_unique_id.clear();
      faces_nb_node.resize(count);
      faces_nodes_unique_id.clear();
      Integer nodeCount = 0;
      for (Integer p = 0; p < count; ++p) {
        ++ptr;
        if (is_3d) {
          nodeCount = CheckedConvert::toInteger(*ptr++);
        }
        else
          nodeCount = 2;
        Integer n = nodeCount;
        faces_nb_node[p] = n;
        orig_nodes_id.resize(n);
        face_nodes_index.resize(n);
        for (Integer z = 0; z < nodeCount; ++z) {
          Integer node_local_id = CheckedConvert::toInteger(*ptr++);
          //info() << "P=" << p << " z=" << z << " id=" << node_local_id;
          orig_nodes_id[z] = node_local_id - 1;
        }

        mesh_utils::reorderNodesOfFace2(orig_nodes_id, face_nodes_index);
        for (Integer z = 0; z < n; ++z)
          faces_nodes_unique_id.add(orig_nodes_id[face_nodes_index[z]]);
        faces_first_node_unique_id.add(orig_nodes_id[face_nodes_index[0]]);
      }

      Int32UniqueArray faces_first_node_local_id(count);
      mesh->nodeFamily()->itemsUniqueIdToLocalId(faces_first_node_local_id, faces_first_node_unique_id, false);

      Integer faces_nodes_unique_id_index = 0;
      for (Integer i_face = 0; i_face < count; ++i_face) {
        Integer n = faces_nb_node[i_face];
        Int64ConstArrayView face_nodes_id(n, &faces_nodes_unique_id[faces_nodes_unique_id_index]);
        Int32 first_node_id = faces_first_node_local_id[i_face];
        // Vérifie que le noeud est dans mon sous-domaine
        if (first_node_id != NULL_ITEM_ID) {
          Node current_node(mesh_nodes[first_node_id]);
          Face face = mesh_utils::getFaceFromNodesUnique(current_node, face_nodes_id);

          // Vérifie que la face est dans mon sous-domaine
          if (!face.null()) {
            faces_local_id.add(current_nb_face + i_face, face.localId());
            ++nb_lima_face_in_sub_domain;
          }
        }
        faces_nodes_unique_id_index += n;
      }
      current_nb_face += count;
    }
    info() << "NB LIMA FACE IN SUB-DOMAIN =" << nb_lima_face_in_sub_domain;
  }

  this->_createGroups(mesh, IK_Face, wrapper.faceGroupReader(), &faces_local_id);

  {
    // Remplit la variable contenant les coordonnées des noeuds
    VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
    ENUMERATE_NODE (i, mesh->ownNodes()) {
      const Node& node = *i;
      nodes_coord_var[node] = nodes_coords.lookupValue(node.uniqueId().asInt64());
      //info() << "Coord: " << node.uniqueId() << " v=" << nodes_coord_var[node];
    }
    nodes_coord_var.synchronize();
  }

  logdate() << "Fin de lecture du fichier";
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LimaMaliReader>
void LimaMalippMeshBase<LimaMaliReader>::
_setCoords(Real3ConstArrayView coords, HashTableMapT<Int64, Real3>& nodes_coords,
           Int64 current_unique_id)
{
  for (Integer i = 0, is = coords.size(); i < is; ++i) {
    Int64 uid = current_unique_id + i;
    HashTableMapT<Int64, Real3>::Data* d = nodes_coords.lookup(uid);
    if (d)
      d->value() = coords[i];
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LimaMaliReader>
void LimaMalippMeshBase<LimaMaliReader>::
_createGroups(IMesh* mesh, eItemKind item_kind, LimaGroupReader<LimaMaliReader>* lima_group_reader,
              HashTableMapT<Int64, Int32>* converter)
{
  IParallelMng* pm = mesh->parallelMng();
  Integer master_rank = pm->masterIORank();
  Integer is_master_io = pm->isMasterIO();
  //IItemFamily* item_family = mesh->itemFamily(item_kind);
  //ItemInternalList cells(cell_family->itemsInternal());

  Int64UniqueArray unique_ids;

  // Création des groupes de mailles
  if (is_master_io) {
    StringUniqueArray groups_name(lima_group_reader->groupsName());
    //std::vector<std::string> groups = wrapper.cellGroups();
    Integer nb_group = groups_name.size();
    Integer sizes_infos[1];
    sizes_infos[0] = nb_group;
    pm->broadcast(IntegerArrayView(1, sizes_infos), master_rank);
    for (Integer i = 0; i < nb_group; ++i) {
      //typename ReaderWrapper::LimaCellGroup group_reader = wrapper.cellGroup(groups[i]);
      String group_name = groups_name[i];
      pm->broadcastString(group_name, master_rank);
      //pwarning() << "Group Name <" << group_name;
      //wrapper.readGroup(group_reader,unique_ids);
      lima_group_reader->read(group_name, unique_ids);
      Integer nb_item_in_group = unique_ids.size();
      sizes_infos[0] = nb_item_in_group;
      pm->broadcast(IntegerArrayView(1, sizes_infos), master_rank);
      pm->broadcast(unique_ids, master_rank);
      if (converter)
        _createGroupFromHashTable(mesh, group_name, item_kind, unique_ids, *converter);
      else
        _createGroupFromUniqueIds(mesh, group_name, item_kind, unique_ids);
    }
  }
  else {
    Integer sizes_infos[1];
    pm->broadcast(IntegerArrayView(1, sizes_infos), master_rank);
    Integer nb_group = sizes_infos[0];
    for (Integer i = 0; i < nb_group; ++i) {
      String group_name;
      pm->broadcastString(group_name, master_rank);
      //pwarning() << "Group Name <" << group_name;
      pm->broadcast(IntegerArrayView(1, sizes_infos), master_rank);
      Integer nb_item_in_group = sizes_infos[0];
      unique_ids.resize(nb_item_in_group);
      pm->broadcast(unique_ids, master_rank);
      if (converter)
        _createGroupFromHashTable(mesh, group_name, item_kind, unique_ids, *converter);
      else
        _createGroupFromUniqueIds(mesh, group_name, item_kind, unique_ids);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LimaMaliReader>
void LimaMalippMeshBase<LimaMaliReader>::
_createGroupFromUniqueIds(IMesh* mesh, const String& name, eItemKind ik,
                          Int64ConstArrayView unique_ids)
{
  Integer nb_item_in_group = unique_ids.size();
  Int32UniqueArray local_ids(nb_item_in_group);
  IItemFamily* family = mesh->itemFamily(ik);
  family->itemsUniqueIdToLocalId(local_ids, unique_ids, false);
  Int32UniqueArray group_ids;
  for (Integer i = 0; i < nb_item_in_group; ++i) {
    if (local_ids[i] != NULL_ITEM_ID)
      group_ids.add(local_ids[i]);
  }
  info() << "Group Name <" << name << "> (" << nb_item_in_group << " elements)";
  LimaUtils::createGroup(family, name, group_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename LimaMaliReader>
void LimaMalippMeshBase<LimaMaliReader>::
_createGroupFromHashTable(IMesh* mesh, const String& name, eItemKind ik,
                          Int64ConstArrayView unique_ids,
                          const HashTableMapT<Int64, Int32>& converter)
{
  Integer nb_item_in_group = unique_ids.size();
  Int32UniqueArray local_ids(nb_item_in_group);
  IItemFamily* family = mesh->itemFamily(ik);
  Int32UniqueArray group_ids;
  for (Integer i = 0; i < nb_item_in_group; ++i) {
    const HashTableMapT<Int64, Int32>::Data* data = converter.lookup(unique_ids[i]);
    if (data)
      group_ids.add(data->value());
  }
  info() << "Group Name <" << name << "> (" << nb_item_in_group << " elements)";
  LimaUtils::createGroup(family, name, group_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
