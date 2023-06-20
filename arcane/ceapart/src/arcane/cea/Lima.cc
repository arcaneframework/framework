// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* Lima.cc                                                     (C) 2000-2023 */
/*                                                                           */
/* Lecture/Ecriture d'un fichier au format Lima.                             */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Iostream.h"
#include "arcane/utils/StdHeader.h"
#include "arcane/utils/HashTableMap.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/ScopedPtr.h"
#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/Collection.h"
#include "arcane/utils/Enumerator.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/Real3.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/StringBuilder.h"

#include "arcane/IMeshReader.h"
#include "arcane/IGhostLayerMng.h"
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
#include "arcane/IMeshWriter.h"
#include "arcane/AbstractService.h"
#include "arcane/ICaseDocument.h"
#include "arcane/ICaseMeshReader.h"
#include "arcane/IMeshBuilder.h"

#include "arcane/cea/LimaCutInfosReader.h"

#include <Lima/lima++.h>

#include <memory>
#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace LimaUtils
{
void createGroup(IItemFamily* family,const String& name,Int32ArrayView local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur des fichiers de maillage via la bibliothèque LIMA.
 */
class LimaMeshBase
: public TraceAccessor
{
 public:

  LimaMeshBase(ISubDomain* sub_domain)
  : TraceAccessor(sub_domain->traceMng()), m_sub_domain(sub_domain) {}
  virtual ~LimaMeshBase() {}

 public:

  virtual bool readMesh(Lima::Maillage& lima,IPrimaryMesh* mesh,const String& filename,
                        const String& dir_name,bool use_internal_partition,Real length_multiplier) =0;

  ISubDomain* subDomain() const { return m_sub_domain; }

 protected:
  
 private:
  
  ISubDomain* m_sub_domain;

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
template<typename ReaderWrapper>
class LimaWrapper
: public LimaMeshBase
{
 public:

  LimaWrapper(ISubDomain* sub_domain)
  : LimaMeshBase(sub_domain), m_cut_infos_reader(new LimaCutInfosReader(sub_domain->parallelMng())) {}

  ~LimaWrapper()
  {
    delete m_cut_infos_reader;
  }

 public:

  virtual bool readMesh(Lima::Maillage& lima,IPrimaryMesh* mesh,const String& filename,
												const String& dir_name,bool use_internal_partition,Real length_multiplier);

 private:
  
  LimaCutInfosReader* m_cut_infos_reader;
	ReaderWrapper m_wrapper;

  bool _readMesh(Lima::Maillage& lima,IPrimaryMesh* mesh,const String& filename,
                 const String& dir_name,bool use_internal_partition,Real length_multiplier);

  void _getProcList(UniqueArray<Integer>& proc_list,const String& dir_name);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LimaMeshReaderWrapper
{
 public:

  void setLima(const Lima::Maillage& lima_mesh)
	{
		m_lima_mesh = lima_mesh;
	}

 protected:

  Lima::Maillage m_lima_mesh;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Lima2DReaderWrapper
: public LimaMeshReaderWrapper
{
 public:
	typedef Lima::Surface LimaCellGroup;
	typedef Lima::Polygone LimaCell;
	typedef Lima::Ligne LimaFaceGroup;
	typedef Lima::Bras LimaFace;
 public:
	Integer nbCellGroup()
	{
		return CheckedConvert::toInteger(m_lima_mesh.nb_surfaces());
	}
	Integer nbFaceGroup()
	{
		return CheckedConvert::toInteger(m_lima_mesh.nb_lignes());
	}
	LimaCellGroup cellGroup(Integer i)
	{
		return m_lima_mesh.surface(i);
	}
	LimaFaceGroup faceGroup(Integer i)
	{
		return m_lima_mesh.ligne(i);
	}
	Integer faceGroupNbFace(const LimaFaceGroup& group)
	{
		return CheckedConvert::toInteger(group.nb_bras());
	}
	LimaFace faceFaceGroup(const LimaFaceGroup& group,Integer i)
	{
		return group.bras(i);
	}
	Integer cellGroupNbCell(const LimaCellGroup& group)
	{
		return CheckedConvert::toInteger(group.nb_polygones());
	}
	LimaCell cellCellGroup(const LimaCellGroup& group,Integer i)
	{
		return group.polygone(i);
	}
	LimaCell cell(Integer i)
	{
		return m_lima_mesh.polygone(i);
	}
	LimaFace face(Integer i)
	{
		return m_lima_mesh.bras(i);
	}
	Integer nbCell()
	{
		return CheckedConvert::toInteger(m_lima_mesh.nb_polygones());
	}
	Integer nbFace()
	{
		return CheckedConvert::toInteger(m_lima_mesh.nb_bras());
	}
	int limaDimension()
	{
		return Lima::D2;
	}
	const char* strDimension()
	{
		return "2D";
	}
	static Integer cellToType(Integer nb_node)
	{
		switch(nb_node){
    case 3: return IT_Triangle3;
    case 4: return IT_Quad4;
    case 5: return IT_Pentagon5;
    case 6: return IT_Hexagon6;
    default:
      break;
    }
		return IT_NullType;
	}
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class Lima3DReaderWrapper
: public LimaMeshReaderWrapper
{
 public:
	typedef Lima::Volume LimaCellGroup;
	typedef Lima::Polyedre LimaCell;
	typedef Lima::Surface LimaFaceGroup;
	typedef Lima::Polygone LimaFace;

	Integer nbCellGroup()
	{
		return CheckedConvert::toInteger(m_lima_mesh.nb_volumes());
	}
	Integer nbFaceGroup()
	{
		return CheckedConvert::toInteger(m_lima_mesh.nb_surfaces());
	}
	LimaCellGroup cellGroup(Integer i)
	{
		return m_lima_mesh.volume(i);
	}
	LimaFaceGroup faceGroup(Integer i)
	{
		return m_lima_mesh.surface(i);
	}
	Integer faceGroupNbFace(const LimaFaceGroup& group)
	{
		return CheckedConvert::toInteger(group.nb_polygones());
	}
	LimaFace faceFaceGroup(const LimaFaceGroup& group,Integer i)
	{
		return group.polygone(i);
	}
	Integer cellGroupNbCell(const LimaCellGroup& group)
	{
		return CheckedConvert::toInteger(group.nb_polyedres());
	}
	LimaCell cellCellGroup(const LimaCellGroup& group,Integer i)
	{
		return group.polyedre(i);
	}
	LimaCell cell(Integer i)
	{
		return m_lima_mesh.polyedre(i);
	}
	LimaFace face(Integer i)
	{
		return m_lima_mesh.polygone(i);
	}
	Integer nbCell()
	{
		return CheckedConvert::toInteger(m_lima_mesh.nb_polyedres());
	}
	Integer nbFace()
	{
		return CheckedConvert::toInteger(m_lima_mesh.nb_polygones());
	}
	int limaDimension()
	{
		return Lima::D3;
	}
	const char* strDimension()
	{
		return "3D";
	}
	static Integer cellToType(Integer nb_node)
	{
		switch(nb_node){
    case 4: return IT_Tetraedron4;
    case 5: return IT_Pyramid5;
    case 6: return IT_Pentaedron6;
    case 8: return IT_Hexaedron8;
    case 10: return IT_Heptaedron10;
    case 12: return IT_Octaedron12;
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
class LimaMeshReader
: public TraceAccessor
{
 public:

  LimaMeshReader(ISubDomain* sd)
  : TraceAccessor(sd->traceMng()), m_sub_domain(sd){}

 public:

  auto readMesh(IPrimaryMesh* mesh, const String& file_name,
                const String& dir_name,bool use_internal_partition,
                bool use_length_unit) -> IMeshReader::eReturnType;

 private:

  ISubDomain* m_sub_domain;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Lecteur des fichiers de maillage via la bibliothèque LIMA.
 */
class LimaMeshReaderService
: public AbstractService
, public IMeshReader
{
 public:

  explicit LimaMeshReaderService(const ServiceBuildInfo& sbi);

 public:
	
	void build() {}

 public:
	
	bool allowExtension(const String& str) override
	{
    return str=="unf" || str=="mli" || str=="mli2" || str=="ice" || str=="uns" || str=="unv";
	}

  eReturnType readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,
                               const String& file_name,
                               const String& dir_name,bool use_internal_partition) override;
  ISubDomain* subDomain() { return m_sub_domain; }

 private:

  ISubDomain* m_sub_domain;

 private:
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(LimaMeshReaderService,
                        ServiceProperty("Lima",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

LimaMeshReaderService::
LimaMeshReaderService(const ServiceBuildInfo& sbi)
: AbstractService(sbi)
, m_sub_domain(sbi.subDomain())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#ifdef ARCANE_LIMA_HAS_MLI
extern "C++" IMeshReader::eReturnType
_directLimaPartitionMalipp(ITimerMng* timer_mng,IPrimaryMesh* mesh,
                           const String& filename,Real length_multiplier);
#endif

#ifdef ARCANE_LIMA_HAS_MLI2
extern "C++" IMeshReader::eReturnType
_directLimaPartitionMalipp2(ITimerMng* timer_mng,IPrimaryMesh* mesh,
                           const String& filename,Real length_multiplier);
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType LimaMeshReaderService::
readMeshFromFile(IPrimaryMesh* mesh,const XmlNode& mesh_node,
                 const String& filename,const String& dir_name,
                 bool use_internal_partition)
{
  ISubDomain* sd = subDomain();

  String case_doc_lang = "en";
  ICaseDocument* case_doc = sd->caseDocument();
  if (case_doc)
    case_doc_lang = case_doc->language();

  // Regarde si on souhaite utiliser l'unité de longueur dans le fichier de maillage.
  String use_unit_attr_name = "utilise-unite";
  String use_unit_str = mesh_node.attrValue(use_unit_attr_name);
  if (case_doc_lang=="en"){
    // Pour des raisons de compatiblité, regarde si 'utilise-unite' est présent
    // auquel cas on le prend. Sinon, on prend le terme anglais.
    if (!use_unit_str.empty()){
      warning() << "'utilise-unite' ne doit être utilisé que pour les JDD en francais."
                << "Utilisez 'use-unit' à la place";
    }
    use_unit_attr_name = "use-unit";
  }
  else{
    // Si non anglais et que 'utilise-unite' n'est pas trouvé,
    // essaie d'utiliser 'use-unit'. Cela est nécessaire pour prendre en compte
    // MeshReaderMng::isUseMeshUnit().
    if (use_unit_str.null()){
      info() << "Attribute '" << use_unit_attr_name << "' is not found. Trying with 'use-unit'";
      use_unit_attr_name = "use-unit";
    }
  }

  // Depuis la 2.8.0 de Arcane, on lit par défaut les unités de longueur si
  // si la variable d'environnement ARCANE_LIMA_DEFAULT_NO_UNIT est définie.
  bool use_length_unit = true;
  if (!platform::getEnvironmentVariable("ARCANE_LIMA_DEFAULT_NO_UNIT").null())
    use_length_unit = false;
  info() << "Default value for unit usage: " << use_length_unit;

  if (use_unit_str.empty())
    use_unit_str = mesh_node.attrValue(use_unit_attr_name);
  info() << "Checking for attribute '" << use_unit_attr_name << "' value='" << use_unit_str << "'";
  if (!use_unit_str.empty()){
    if (use_unit_str=="1" || use_unit_str=="true")
      use_length_unit = true;
    else if (use_unit_str=="0" || use_unit_str=="false")
      use_length_unit = false;
    else
      ARCANE_FATAL("Invalid value boolean value '{0}' for '{1}' attribute."
                   " Valid values are '0', '1' 'true' or 'false'",
                   use_unit_str,use_unit_attr_name);
  }

  info() << "Utilise l'unité de longueur de Lima: " << use_length_unit << " (lang=" << case_doc_lang << ")";

  LimaMeshReader reader(sd);
  return reader.readMesh(mesh,filename,dir_name,use_internal_partition,use_length_unit);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IMeshReader::eReturnType LimaMeshReader::
readMesh(IPrimaryMesh* mesh,const String& filename,const String& dir_name,
         bool use_internal_partition, bool use_length_unit)
{
  if (filename.null() || filename.empty())
    return IMeshReader::RTIrrelevant;

  ISubDomain* sd = m_sub_domain;
  ITimerMng* timer_mng = sd->timerMng();
	LimaMeshBase* lm = 0;
  ICaseDocument* case_doc = sd->caseDocument();

  info() << "Lima: use_length_unit=" << use_length_unit
         << " use_internal_partition=" << use_internal_partition;
  Real length_multiplier = 0.0;
  if (use_length_unit){
    String code_system;
    if (case_doc)
      code_system = case_doc->codeUnitSystem();
    if (code_system.null() || code_system.empty()){
      info() << "No unit system configured. Use MKS unit system.";
      length_multiplier = 1.0;
    }
    else if (code_system=="CGS"){
      length_multiplier = 100.0;
    }
    else if (code_system=="MKS"){
      length_multiplier = 1.0;
    }
    else{
      ARCANE_FATAL("Unknown unit system '{0}' (valid values are: 'CGS' ou 'MKS'",code_system);
    }
  }

  std::string loc_file_name = filename.localstr();
  size_t rpos = loc_file_name.rfind(".mli");
  size_t rpos2 = loc_file_name.rfind(".mli2");
  info() << " FILE_NAME=" << loc_file_name;
  info() << " RPOS MLI=" << rpos << " s=" << loc_file_name.length();
  info() << " RPOS MLI2=" << rpos2 << " s=" << loc_file_name.length();
  //TODO mettre has_thread = true que si on utilise le gestionnaire de parallelisme
  // via les threads
  bool has_thread = false; //arcaneHasThread();
  // On ne peut pas utiliser l'api mali pp avec les threads
  if (!has_thread && use_internal_partition && ((rpos+4)==loc_file_name.length())){
    info() << "Use direct partitioning with mli";
#ifdef ARCANE_LIMA_HAS_MLI
    return _directLimaPartitionMalipp(timer_mng,mesh,filename,length_multiplier);
#else
    ARCANE_FATAL("Can not use 'mli' files because Lima is not compiled with 'mli' support");
#endif
  }
  else if (!has_thread && use_internal_partition && ((rpos2+5)==loc_file_name.length())){
    info() << "Use direct partitioning with mli2";
#ifdef ARCANE_LIMA_HAS_MLI2
    return _directLimaPartitionMalipp2(timer_mng,mesh,filename,length_multiplier);
#else
    ARCANE_FATAL("Can not use 'mli2' files because Lima is not compiled with 'mli2' support");
#endif
  }
  else {
    info() << "Chargement Lima du fichier '" << filename << "'";

    const char* version = Lima::lima_version();
    info() << "Utilisation de la version " << version << " de Lima";

    Timer time_to_read(sd,"ReadLima",Timer::TimerReal);

    // Aucune préparation spécifique à faire
    LM_TYPEMASQUE preparation = LM_ORIENTATION | LM_COMPACTE;

    log() << "Début lecture fichier " << filename;
  
    Lima::Maillage lima(filename.localstr());

    try{
      {
        Timer::Sentry sentry(&time_to_read);
        Timer::Phase t_action(sd,TP_InputOutput);
        lima.lire(filename.localstr(),Lima::SUFFIXE,true);
        //warning() << "Preparation lima supprimée";
        lima.preparation_parametrable(preparation);
      }
    }
    catch(const Lima::erreur& ex){
      ARCANE_FATAL("Can not read lima file '{0}' error is '{1}'",filename,ex.what());
    }
    catch(...){
      ARCANE_FATAL("Can not read lima file '{0}'",filename);
    }
    
    info() << "Temps de lecture et préparation du maillage (unité: seconde): "
           << time_to_read.lastActivationTime();
    // Si la dimension n'est pas encore positionnée, utilise celle
    // donnée par Lima.
    if (mesh->dimension()<=0){
      if (lima.dimension()==Lima::D3){
        mesh->setDimension(3);
        info() << "Maillage 3D";
      }
      else if (lima.dimension()==Lima::D2){
        mesh->setDimension(2);
        info() << "Maillage 2D";
      }
    }

    if (mesh->dimension()==3){
      lm = new LimaWrapper<Lima3DReaderWrapper>(sd);
    }
    else if (mesh->dimension()==2){
      lm = new LimaWrapper<Lima2DReaderWrapper>(sd);
    }
    if (!lm){
      log() << "Dimension du maillage non reconnue par lima";
      return IMeshReader::RTIrrelevant;
    }
    
    bool ret = lm->readMesh(lima,mesh,filename,dir_name,use_internal_partition,length_multiplier);
    if (ret)
      return IMeshReader::RTError;

    // A faire si plusieurs couches de mailles fantomes
    {
      Integer nb_ghost_layer = mesh->ghostLayerMng()->nbGhostLayer();
      if (nb_ghost_layer>1)
        mesh->synchronizeGroupsAndVariables();
    }

    delete lm;
  }

  return IMeshReader::RTOk;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ReaderWrapper> bool LimaWrapper<ReaderWrapper>::
readMesh(Lima::Maillage& lima,IPrimaryMesh* mesh,const String& filename,
         const String& dir_name,bool use_internal_partition,Real length_multiplier)
{
  return _readMesh(lima,mesh,filename,dir_name,use_internal_partition,length_multiplier);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ReaderWrapper> bool LimaWrapper<ReaderWrapper>::
_readMesh(Lima::Maillage& lima,IPrimaryMesh* mesh,const String& file_name,
          const String& dir_name,bool use_internal_partition,Real length_multiplier)
{
  ARCANE_UNUSED(file_name);

  // Il faut utiliser le parallelMng du maillage qui peut être différent
  // de celui du sous-domaine, par exemple si le maillage est séquentiel.
  IParallelMng* pm = mesh->parallelMng();

  bool is_parallel = pm->isParallel();
  Integer sid = pm->commRank();

  Integer mesh_nb_node = 0;
  Integer nb_edge = 0;
  Integer lima_nb_face = 0;
  Integer mesh_nb_cell = 0;

  m_wrapper.setLima(lima);

  mesh_nb_node = CheckedConvert::toInteger(lima.nb_noeuds());
  mesh_nb_cell = m_wrapper.nbCell(); //lima.nb_polyedres();
  lima_nb_face = m_wrapper.nbFace(); //lima.nb_polygones();
  nb_edge = 0; //lima.nb_bras();

  info() << "-- Informations sur le maillage (Interne):";
  info() << "Nombre de noeuds  " << mesh_nb_node;
  info() << "Nombre d'arêtes   " << nb_edge;
  info() << "Nombre de faces   " << lima_nb_face;
  info() << "Nombre de mailles " << mesh_nb_cell;
  info() << "-- Informations sur le maillage (Lima):";
  info() << "Nombre de noeuds    " << lima.nb_noeuds();
  info() << "Nombre d'arêtes     " << lima.nb_bras();
  info() << "Nombre de polygones " << lima.nb_polygones();
  info() << "Nombre de polyedres " << lima.nb_polyedres();
  info() << "Nombre de surfaces  " << lima.nb_surfaces();
  info() << "Nombre de volumes   " << lima.nb_volumes();

  info() << "Unité de longueur du fichier: " << lima.unite_longueur();
  // Si 0.0, indique qu'on ne souhaite pas utiliser l'unité du fichier.
  // Dans ce, cela correspond à un multiplicateur de 1.0
  if (length_multiplier==0.0)
    length_multiplier = 1.0;
  else
    length_multiplier *= lima.unite_longueur();

  if (mesh_nb_node==0){
    ARCANE_FATAL("No node in mesh");
  }

  // ------------------------------------------------------------
  //    -------------------------------- Création des mailles
  // ------------------------------------------------------------


  // Lit les numéros uniques des entités (en parallèle)
  UniqueArray<Int64> nodes_unique_id(mesh_nb_node);
  UniqueArray<Int64> cells_unique_id(mesh_nb_cell);
  if (is_parallel && !use_internal_partition && pm->commSize()>1){
    m_cut_infos_reader->readItemsUniqueId(nodes_unique_id,cells_unique_id,dir_name);
  }
  else{
    for( Integer i=0; i<mesh_nb_node; ++i )
      nodes_unique_id[i] = i;
    for( Integer i=0; i<mesh_nb_cell; ++i )
      cells_unique_id[i] = i;
  }

  // Pour l'instant, laisse à false.
  // Si true, les uid sont incrémentés pour commencer à un.
  bool first_uid_is_one = false;
  if (!platform::getEnvironmentVariable("ARCANE_LIMA_UNIQUE_ID").null()){
    first_uid_is_one = true;
    info() << "WARNING: UniqueId begin at 1";
  }
  if (first_uid_is_one){
    for( Integer i=0; i<mesh_nb_node; ++i )
      ++nodes_unique_id[i];
    for( Integer i=0; i<mesh_nb_cell; ++i )
      ++cells_unique_id[i];
  }

  HashTableMapT<Int64,Real3> nodes_coords(mesh_nb_node,true);
  if (!math::isEqual(length_multiplier,1.0)){
    info() << "Using length multiplier v=" << length_multiplier;
    for( Integer i=0; i<mesh_nb_node; ++i ){
      const Lima::Noeud& node = lima.noeud(i);
      Real3 coord(node.x(),node.y(),node.z());
      coord *= length_multiplier;
      nodes_coords.nocheckAdd(nodes_unique_id[i],coord);
    }
  }
  else{
    for( Integer i=0; i<mesh_nb_node; ++i ){
      const Lima::Noeud& node = lima.noeud(i);
      Real3 coord(node.x(),node.y(),node.z());
      nodes_coords.nocheckAdd(nodes_unique_id[i],coord);
    }
  }

  Integer cells_infos_index = 0;

  UniqueArray<Integer> cells_filter;

  bool use_own_mesh = false;

  if (is_parallel && pm->commSize()>1){
    use_own_mesh = true;
    if (use_internal_partition)
      use_own_mesh = false;
  }

  typedef typename ReaderWrapper::LimaCellGroup LimaCellGroup;
  typedef typename ReaderWrapper::LimaCell LimaCell;
  typedef typename ReaderWrapper::LimaFaceGroup LimaFaceGroup;
  typedef typename ReaderWrapper::LimaFace LimaFace;

	if (use_own_mesh){
    Integer nb = m_wrapper.nbCellGroup();
    for( Integer i=0; i<nb; ++i ){
      const LimaCellGroup& lima_group = m_wrapper.cellGroup(i);
      std::string group_name = lima_group.nom();
      if (group_name=="LOCAL" || group_name=="local"){
        Integer nb_own_cell = m_wrapper.cellGroupNbCell(lima_group);
        cells_filter.resize(nb_own_cell);
        for( Integer z=0; z<nb_own_cell; ++z ){
          cells_filter[z] = CheckedConvert::toInteger(m_wrapper.cellCellGroup(lima_group,z).id() - 1);
        }
      }
    }
  }
  else{
    if (use_internal_partition && sid!=0){
      mesh_nb_cell = 0;
      mesh_nb_node = 0;
      lima_nb_face = 0;
    }
    cells_filter.resize(mesh_nb_cell);
    for( Integer i=0; i<mesh_nb_cell; ++i )
      cells_filter[i] = i;
  }

  // Calcul le nombre de mailles/noeuds
  Integer mesh_nb_cell_node = 0;
  for( Integer j=0, js=cells_filter.size(); j<js; ++j ){
    mesh_nb_cell_node += CheckedConvert::toInteger(m_wrapper.cell(cells_filter[j]).nb_noeuds());
  }
  
  // Tableau contenant les infos aux mailles (voir IMesh::allocateMesh())
  UniqueArray<Int64> cells_infos(mesh_nb_cell_node+cells_filter.size()*2);

  // Remplit le tableau contenant les infos des mailles
  for( Integer i_cell=0, s_cell=cells_filter.size(); i_cell<s_cell; ++i_cell ){

    Integer cell_indirect_id = cells_filter[i_cell];
    LimaCell lima_cell = m_wrapper.cell(cell_indirect_id);
    Integer n = CheckedConvert::toInteger(lima_cell.nb_noeuds());

    Integer ct = ReaderWrapper::cellToType(n);
    if (ct==IT_NullType)
      throw UnknownItemTypeException("Lima::readFile: Cell",n,cell_indirect_id);
    // Stocke le type de la maille
    cells_infos[cells_infos_index] = ct;
    ++cells_infos_index;
    // Stocke le numéro unique de la maille
    cells_infos[cells_infos_index] = cells_unique_id[cell_indirect_id];
    ++cells_infos_index;

    // Rempli la liste des numéros des noeuds de la maille
    for( Integer z=0, sz=n; z<sz; ++z ){
      Int64 node_uid = nodes_unique_id[CheckedConvert::toInteger(lima_cell.noeud(z).id()-1)];
      cells_infos[cells_infos_index+z] = node_uid;
    }

#if 0
    cout << "CELL LIMA1 " << cells_unique_id[cell_indirect_id] << " ";
    for( Integer z=0; z<n; ++z )
      cout << " " << cells_infos[cells_infos_index+z];
    cout << '\n';
#endif

    cells_infos_index += n;
  }

  logdate() << "Début allocation du maillage nb_cell=" << cells_filter.size();
  mesh->allocateCells(cells_filter.size(),cells_infos,false);
  logdate() << "Fin allocation du maillage";

  // Positionne les propriétaires des noeuds à partir des groupes de noeuds de Lima
  if (use_internal_partition){
    ItemInternalList nodes(mesh->itemsInternal(IK_Node));
    for( Integer i=0, is=nodes.size(); i<is; ++i )
      nodes[i]->setOwner(sid,sid);
    ItemInternalList cells(mesh->itemsInternal(IK_Cell));
    for( Integer i=0, is=cells.size(); i<is; ++i )
      cells[i]->setOwner(sid,sid);
  }
  else{
    Int64UniqueArray unique_ids;
    Int32UniqueArray local_ids;
    Integer sub_domain_id = subDomain()->subDomainId();
    Integer nb = CheckedConvert::toInteger(lima.nb_nuages());
    for( Integer i=0; i<nb; ++i ){
      const Lima::Nuage& lima_group = lima.nuage(i);
      Integer nb_item_in_group = CheckedConvert::toInteger(lima_group.nb_noeuds());
      std::string group_name = lima_group.nom();
      unique_ids.resize(nb_item_in_group);
      local_ids.resize(nb_item_in_group);
      for( Integer z=0; z<nb_item_in_group; ++z ){
        unique_ids[z] = nodes_unique_id[CheckedConvert::toInteger(lima_group.noeud(z).id() - 1)];
      }
      mesh->nodeFamily()->itemsUniqueIdToLocalId(local_ids,unique_ids,false);
      bool remove_group = false;
      if (group_name=="LOCALN" || group_name=="localn"){
        info() << "Utilisation du groupe 'LOCALN' pour indiquer que les "
               << "noeuds appartiennent au sous-domaine";
        ItemInternalList nodes(mesh->itemsInternal(IK_Node));
        for( Integer z=0, sz=nb_item_in_group; z<sz; ++z ){
          Integer local_id = local_ids[z];
          if (local_id!=NULL_ITEM_ID)
            nodes[local_id]->setOwner(sub_domain_id,sub_domain_id);
        }
        remove_group = true;
      }
      debug() << "Vérification du groupe '" << group_name << "'";
      if (group_name.length()>3 && !remove_group){
        String grp = group_name; //.c_str();
        if (grp.startsWith("NF_")){
          grp = grp.substring(3);
          Int32 ghost_sub_domain_id = 0;
          bool is_bad = builtInGetValue(ghost_sub_domain_id,grp);
          debug() << "Vérification du groupe '" << group_name << "' (3) " << is_bad;
          if (!is_bad){
            info() << "Utilisation du groupe " << group_name << " pour indiquer que le "
                   << "sous-domaine " << ghost_sub_domain_id << " est propriétaire de ses noeuds";
            ItemInternalList nodes(mesh->itemsInternal(IK_Node));
            for( Integer z=0, sz=nb_item_in_group; z<sz; ++z ){
              Integer local_id = local_ids[z];
              if (local_id!=NULL_ITEM_ID)
                nodes[local_ids[z]]->setOwner(ghost_sub_domain_id,sub_domain_id);
            }
            remove_group = true;
          }
        }
      }
    }
  }

  mesh->endAllocate();

  // Comme le maillage créé lui même ses faces sans tenir compte de celles qui
  // existent éventuellement dans Lima, il faut maintenant déterminer le numéro
  // local dans notre maillage de chaque face de Lima.
  UniqueArray<Integer> faces_id(lima_nb_face); // Numéro de la face lima dans le maillage \a mesh
  {
    // Nombre de faces/noeuds
    Integer face_nb_node = 0;
    for( Integer i_face=0; i_face<lima_nb_face; ++i_face ){
      const LimaFace& lima_face = m_wrapper.face(i_face);
      face_nb_node += CheckedConvert::toInteger(lima_face.nb_noeuds());
    }

    UniqueArray<Int64> faces_first_node_unique_id(lima_nb_face);
    UniqueArray<Int32> faces_first_node_local_id(lima_nb_face);
    UniqueArray<Int64> faces_nodes_unique_id(face_nb_node);
    Integer faces_nodes_unique_id_index = 0;

    UniqueArray<Int64> orig_nodes_id;
    orig_nodes_id.reserve(100);
    
    UniqueArray<Integer> face_nodes_index;
    face_nodes_index.reserve(100);
    
    ItemInternalList mesh_nodes(mesh->itemsInternal(IK_Node));
    
    for( Integer i_face=0; i_face<lima_nb_face; ++i_face ){
      const LimaFace& lima_face = m_wrapper.face(i_face);
      Integer n = CheckedConvert::toInteger(lima_face.nb_noeuds());
      orig_nodes_id.resize(n);
      face_nodes_index.resize(n);
      for( Integer z=0; z<n; ++z )
        orig_nodes_id[z] = nodes_unique_id[CheckedConvert::toInteger(lima_face.noeud(z).id() - 1)];
      //face_orig_nodes_id[z] = lima_face.noeud(z).id() - 1;

#if 0
      cout << "FACE LIMA1 " << lima_face.id()-1 << " ";
      for( Integer z=0; z<n; ++z )
        cout << " " << orig_nodes_id[z];
      cout << '\n';
#endif

      mesh_utils::reorderNodesOfFace2(orig_nodes_id,face_nodes_index);
      for( Integer z=0; z<n; ++z )
        faces_nodes_unique_id[faces_nodes_unique_id_index+z] = orig_nodes_id[face_nodes_index[z]];
      faces_first_node_unique_id[i_face] = orig_nodes_id[face_nodes_index[0]];
      faces_nodes_unique_id_index += n;
    }

    mesh->nodeFamily()->itemsUniqueIdToLocalId(faces_first_node_local_id,faces_first_node_unique_id);

    faces_nodes_unique_id_index = 0;
    for( Integer i_face=0; i_face<lima_nb_face; ++i_face ){
      const LimaFace& lima_face = m_wrapper.face(i_face);
      Integer n = CheckedConvert::toInteger(lima_face.nb_noeuds());
      Int64ConstArrayView face_nodes_id(n,&faces_nodes_unique_id[faces_nodes_unique_id_index]);
      Node current_node(mesh_nodes[faces_first_node_local_id[i_face]]);
      Face face = mesh_utils::getFaceFromNodesUnique(current_node,face_nodes_id);

      if (face.null()){
        OStringStream ostr;
        ostr() << "(Nodes:";
        for( Integer z=0; z<n; ++z )
          ostr() << ' ' << face_nodes_id[z];
        ostr() << " - " << current_node.localId() << ")";
        ARCANE_FATAL("INTERNAL: Lima face index={0} with nodes '{1}' is not in node/face connectivity",
                     i_face,ostr.str());
      }
      faces_id[i_face] = face.localId();

      faces_nodes_unique_id_index += n;
    }
  }

  IItemFamily* node_family = mesh->nodeFamily();
  IItemFamily* face_family = mesh->faceFamily();
  IItemFamily* cell_family = mesh->cellFamily();

  // Création des groupes
  if (use_internal_partition && sid!=0){
    {
      Integer nb = CheckedConvert::toInteger(lima.nb_nuages());
      for( Integer i=0; i<nb; ++i ){
        const Lima::Nuage& lima_group = lima.nuage(i);
        std::string group_name = lima_group.nom();
        LimaUtils::createGroup(node_family,group_name,Int32ArrayView());
      }
    }
    {
      Integer nb = m_wrapper.nbFaceGroup();
      for( Integer i=0; i<nb; ++i ){
        const LimaFaceGroup& lima_group = m_wrapper.faceGroup(i);
        std::string group_name = lima_group.nom();
				LimaUtils::createGroup(face_family,group_name,Int32ArrayView());
      }
    }
    {
      Integer nb = m_wrapper.nbCellGroup();
      for( Integer i=0; i<nb; ++i ){
        const LimaCellGroup& lima_group = m_wrapper.cellGroup(i);
        std::string group_name = lima_group.nom();
				LimaUtils::createGroup(cell_family,group_name,Int32ArrayView());
      }
    }
  }
  else{
    UniqueArray<Int64> unique_ids;
    UniqueArray<Int32> local_ids;
    Integer sub_domain_id = subDomain()->subDomainId();
    // Création des groupes de noeuds
    {
      Integer nb = CheckedConvert::toInteger(lima.nb_nuages());
      for( Integer i=0; i<nb; ++i ){
        const Lima::Nuage& lima_group = lima.nuage(i);
        Integer nb_item_in_group = CheckedConvert::toInteger(lima_group.nb_noeuds());
        std::string group_name = lima_group.nom();
        unique_ids.resize(nb_item_in_group);
        local_ids.resize(nb_item_in_group);
        for( Integer z=0; z<nb_item_in_group; ++z ){
          Integer lima_node_id = CheckedConvert::toInteger(lima_group.noeud(z).id());
          unique_ids[z] = nodes_unique_id[lima_node_id - 1];
        }
        mesh->nodeFamily()->itemsUniqueIdToLocalId(local_ids,unique_ids);
        bool remove_group = false;
        if (group_name=="LOCALN" || group_name=="localn"){
          remove_group = true;
        }
        debug() << "Vérification du groupe '" << group_name << "'";
        if (group_name.length()>3 && !remove_group){
          String grp = group_name.c_str();
          //grp.left(3);
          debug() << "Vérification du groupe '" << group_name << "' (2) '" << grp << "'";
          if (grp.startsWith("NF_")){
            //grp = group_name.c_str();
            //grp.right(group_name.length()-3);
            grp = grp.substring(3);
            Integer ghost_sub_domain_id = 0;
            bool is_bad = builtInGetValue(ghost_sub_domain_id,grp);
            debug() << "Vérification du groupe '" << group_name << "' (3) " << is_bad;
            if (!is_bad){
              remove_group = true;
            }
          }
        }
        if (!remove_group){
          log() << "NodeGroup Name <" << group_name << "> (" << nb_item_in_group << " elements)";
          LimaUtils::createGroup(node_family,group_name,local_ids);
        }
      }
    }
    // Création des groupes de faces
    {
      Integer nb = m_wrapper.nbFaceGroup();
      for( Integer i=0; i<nb; ++i ){
        const LimaFaceGroup& lima_group = m_wrapper.faceGroup(i);
        Integer nb_item_in_group = m_wrapper.faceGroupNbFace(lima_group);
        local_ids.resize(nb_item_in_group);
        // Comme les numéros des faces données par Lima ne correspondent
        // pas à celles créées dans Arcane, on utilise le tableau de correspondance
        for( Integer z=0; z<nb_item_in_group; ++z ){
          local_ids[z] = faces_id[CheckedConvert::toInteger(m_wrapper.faceFaceGroup(lima_group,z).id() - 1)];
        }
        std::string group_name = lima_group.nom();
        log() << "FaceGroup Name <" << group_name << "> (" << nb_item_in_group << " elements)";
        LimaUtils::createGroup(face_family,group_name,local_ids);
      }
    }
    // Création des groupes de mailles
    {
      Integer nb = m_wrapper.nbCellGroup();
      for( Integer i=0; i<nb; ++i ){
        const LimaCellGroup& lima_group = m_wrapper.cellGroup(i);
        Integer nb_item_in_group = m_wrapper.cellGroupNbCell(lima_group);
        std::string group_name = lima_group.nom();
        unique_ids.resize(nb_item_in_group);
        local_ids.resize(nb_item_in_group);
        for( Integer z=0; z<nb_item_in_group; ++z ){
          unique_ids[z] = cells_unique_id[CheckedConvert::toInteger(m_wrapper.cellCellGroup(lima_group,z).id() - 1)];
        }
        mesh->cellFamily()->itemsUniqueIdToLocalId(local_ids,unique_ids);
        bool remove_group = false;
        if (group_name=="LOCAL" || group_name=="local"){
          ItemInternalList cells(mesh->itemsInternal(IK_Cell));
          for( Integer z=0, sz=nb_item_in_group; z<sz; ++z )
            cells[local_ids[z]]->setOwner(sub_domain_id,sub_domain_id);
          remove_group = true;
        }
        if (!remove_group){
          String grp(group_name.c_str());
          if (grp.startsWith("MF_")){
            info() << "Le groupe de mailles " << group_name << " n'est pas utilisé";
            remove_group = true;
          }
        }
        if (!remove_group){
          log() << "CellGroup Name <" << group_name << "> (" << nb_item_in_group << " elements)";
          LimaUtils::createGroup(cell_family,group_name,local_ids);
        }
      }
    }
  }


  {
    // Remplit la variable contenant les coordonnées des noeuds
    // En parallèle avec le maillage déjà découpé par Decoupe3D, le
    // maillage contient déjà une couche de mailles fantomes. Pour
    // avoir les coordonnées des noeuds si on a besoin que d'une couche
    // de mailles fantomes, il suffit donc de parcourir tous
    // les noeuds. Si on a besoin de plusieurs couches de mailles fantomes,
    // il faut faire une synchronisation.
    VariableNodeReal3& nodes_coord_var(mesh->nodesCoordinates());
    NodeGroup nodes = mesh->allNodes();
    Integer nb_ghost_layer = mesh->ghostLayerMng()->nbGhostLayer();
    if (nb_ghost_layer>1)
      nodes = mesh->ownNodes();
    ENUMERATE_NODE(i,nodes){
      const Node& node = *i;
      nodes_coord_var[node] = nodes_coords.lookupValue(node.uniqueId());
      //info() << "Coord: " << node.uniqueId() << " v=" << nodes_coord_var[node];
    }
    if (nb_ghost_layer>1)
      nodes_coord_var.synchronize();
  }


  //    -------------------------------- Lecture des groupes
  info() << "Nombre de nuages   " << lima.nb_nuages();
  info() << "Nombre de lignes   " << lima.nb_lignes();
  info() << "Nombre de surfaces " << lima.nb_surfaces();
  info() << "Nombre de volumes  " << lima.nb_volumes();

  //    -------------------------------- Création des structures internes.

  //_buildInternalMesh();

  logdate() << "Fin de lecture du fichier";
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename ReaderWrapper> void LimaWrapper<ReaderWrapper>::
_getProcList(UniqueArray<Integer>& proc_list,const String& dir_name)
{
  ISubDomain* sd = subDomain();
  StringBuilder comm_file_nameb;
  if (!dir_name.empty()){
    comm_file_nameb += dir_name;
    comm_file_nameb += "/";
  }
  comm_file_nameb += "Communications";
  String comm_file_name = comm_file_nameb.toString();

  // Lecture du fichier de communication
  ScopedPtrT<IXmlDocumentHolder> doc_holder(sd->ioMng()->parseXmlFile(comm_file_name));
  if (!doc_holder.get())
    ARCANE_FATAL("Invalid file '{0}'",comm_file_name);

  XmlNode root_elem = doc_holder->documentNode().documentElement();
  XmlNode cpu_elem;
  XmlNodeList cpu_list = root_elem.child(String("cpus")).children(String("cpu-from"));

  String ustr_buf  = String::fromNumber(sd->subDomainId());
  String ustr_id(String("id"));
  for( Integer i=0, s=cpu_list.size(); i<s; ++i ){
    String id_str = cpu_list[i].attrValue(ustr_id);
    if (id_str==ustr_buf){
      cpu_elem = cpu_list[i];
      break;
    }
  }
  if (cpu_elem.null())
    ARCANE_FATAL("No element <cpus/cpu-from[@id=\"{0}\"]>",sd->subDomainId());
  {
    cpu_list = cpu_elem.children(String("cpu-to"));
    debug() << "Nb procs " << cpu_list.size();
    for( Integer i=0; i<cpu_list.size(); ++i ){
      Integer v = cpu_list[i].valueAsInteger();
      proc_list.add(v);
      debug() << "Read proc " << v;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LimaCaseMeshReader
: public AbstractService
, public ICaseMeshReader
{
 public:

  class Builder
  : public IMeshBuilder
  {
   public:

    explicit Builder(ISubDomain* sd, const CaseMeshReaderReadInfo& read_info)
    : m_sub_domain(sd)
    , m_trace_mng(sd->traceMng())
    , m_read_info(read_info)
    {}

   public:

    void fillMeshBuildInfo(MeshBuildInfo& build_info) override
    {
      ARCANE_UNUSED(build_info);
    }
    void allocateMeshItems(IPrimaryMesh* pm) override
    {
      LimaMeshReader reader(m_sub_domain);
      String fname = m_read_info.fileName();
      m_trace_mng->info() << "Lima Reader (ICaseMeshReader) file_name=" << fname;
      bool use_length_unit = true; // Avec le ICaseMeshReader on utilise toujours le système d'unité.
      String directory_name = m_read_info.directoryName();
      IMeshReader::eReturnType ret = reader.readMesh(pm, fname, directory_name, m_read_info.isParallelRead(), use_length_unit);
      if (ret != IMeshReader::RTOk)
        ARCANE_FATAL("Can not read MSH File");
    }

   private:

    ISubDomain* m_sub_domain;
    ITraceMng* m_trace_mng;
    CaseMeshReaderReadInfo m_read_info;
  };

 public:

  explicit LimaCaseMeshReader(const ServiceBuildInfo& sbi)
  : AbstractService(sbi), m_sub_domain(sbi.subDomain())
  {}

 public:

  Ref<IMeshBuilder> createBuilder(const CaseMeshReaderReadInfo& read_info) const override
  {
    IMeshBuilder* builder = nullptr;
    String str = read_info.format();
    if (str=="unf" || str=="mli" || str=="mli2" || str=="ice" || str=="uns" || str=="unv")
      builder = new Builder(m_sub_domain, read_info);
    return makeRef(builder);
  }

 private:

  ISubDomain* m_sub_domain = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(LimaCaseMeshReader,
                        ServiceProperty("LimaCaseMeshReader", ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(ICaseMeshReader));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class LimaMeshWriter
: public AbstractService
, public IMeshWriter
{
 public:
  LimaMeshWriter(const ServiceBuildInfo& sbi)
  : AbstractService(sbi), m_sub_domain(sbi.subDomain()) {}
 public:
  virtual void build() {}
  virtual bool writeMeshToFile(IMesh* mesh,const String& file_name);

 private:
  ISubDomain* m_sub_domain;
  void _writeItem(Lima::Maillage& lima_mesh,ConstArrayView<Lima::Noeud> nodes,
									ItemWithNodes item);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_SERVICE(LimaMeshWriter,
                        ServiceProperty("Lima",ST_SubDomain),
                        ARCANE_SERVICE_INTERFACE(IMeshWriter));

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void LimaMeshWriter::
_writeItem(Lima::Maillage& m,ConstArrayView<Lima::Noeud> nodes,ItemWithNodes c)
{
	switch(c.type()){
	case IT_Octaedron12:
		m.ajouter(Lima::Polyedre(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
														 nodes[c.node(2).localId()],nodes[c.node(3).localId()],
														 nodes[c.node(4).localId()],nodes[c.node(5).localId()],
														 nodes[c.node(6).localId()],nodes[c.node(7).localId()],
														 nodes[c.node(8).localId()],nodes[c.node(9).localId()],
														 nodes[c.node(10).localId()],nodes[c.node(11).localId()]));
		break;
	case IT_Heptaedron10:
		m.ajouter(Lima::Polyedre(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
														 nodes[c.node(2).localId()],nodes[c.node(3).localId()],
														 nodes[c.node(4).localId()],nodes[c.node(5).localId()],
														 nodes[c.node(6).localId()],nodes[c.node(7).localId()],
														 nodes[c.node(8).localId()],nodes[c.node(9).localId()]));
		break;
	case IT_Hexaedron8:
		m.ajouter(Lima::Polyedre(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
														 nodes[c.node(2).localId()],nodes[c.node(3).localId()],
														 nodes[c.node(4).localId()],nodes[c.node(5).localId()],
														 nodes[c.node(6).localId()],nodes[c.node(7).localId()]));
		break;
	case IT_Pentaedron6:
		m.ajouter(Lima::Polyedre(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
														 nodes[c.node(2).localId()],nodes[c.node(3).localId()],
														 nodes[c.node(4).localId()],nodes[c.node(5).localId()]));
		break;
	case IT_Pyramid5:
		m.ajouter(Lima::Polyedre(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
														 nodes[c.node(2).localId()],nodes[c.node(3).localId()],
														 nodes[c.node(4).localId()]));
		break;
	case IT_Tetraedron4:
		m.ajouter(Lima::Polyedre(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
														 nodes[c.node(2).localId()],nodes[c.node(3).localId()]));
		break;
	case IT_Hexagon6:
		m.ajouter(Lima::Polygone(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
																nodes[c.node(2).localId()],nodes[c.node(3).localId()],
																nodes[c.node(4).localId()],nodes[c.node(5).localId()]));
		break;
	case IT_Pentagon5:
		m.ajouter(Lima::Polygone(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
																nodes[c.node(2).localId()],nodes[c.node(3).localId()],
																nodes[c.node(4).localId()]));
		break;
	case IT_Quad4:
		m.ajouter(Lima::Polygone(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
																nodes[c.node(2).localId()],nodes[c.node(3).localId()]));
		break;
	case IT_Triangle3:
		m.ajouter(Lima::Polygone(nodes[c.node(0).localId()],nodes[c.node(1).localId()],
																nodes[c.node(2).localId()]));
		break;
	case IT_Line2:
		m.ajouter(Lima::Bras(nodes[c.node(0).localId()],nodes[c.node(1).localId()]));
		break;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Ecriture du maillage au format Lima.
 *
 * \warning La numérotation des entités doit être contiguee.
 */
bool LimaMeshWriter::
writeMeshToFile(IMesh* mesh,const String& file_name)
{
  ITraceMng* trace = mesh->traceMng();
	int dimension = mesh->dimension();

  std::string std_file_name = file_name.localstr();
	//TODO: FAIRE EXTENSION si non presente
  // Regarde si le fichier a l'extension '.unf' ou '.mli'.
  // Sinon, ajoute '.mli'
  std::string::size_type std_end = std::string::npos;
  if (std_file_name.rfind(".mli")==std_end && std_file_name.rfind(".unf")==std_end){
    std_file_name += ".mli";
  }
  info() << "FINAL_FILE_NAME=" << std_file_name;
  Lima::Maillage lima(std_file_name);

  if (dimension==3)
    lima.dimension(Lima::D3);
  else if (dimension==2)
    lima.dimension(Lima::D2);
	
  IItemFamily* node_family = mesh->nodeFamily();
  IItemFamily* edge_family = mesh->edgeFamily();
  IItemFamily* face_family = mesh->faceFamily();
  IItemFamily* cell_family = mesh->cellFamily();

  Integer mesh_nb_node = node_family->nbItem();
  Integer mesh_nb_edge = edge_family->nbItem();
  Integer mesh_nb_face = face_family->nbItem();
  Integer mesh_nb_cell = cell_family->nbItem();

  NodeInfoListView nodes(node_family);
  EdgeInfoListView edges(edge_family);
  FaceInfoListView faces(face_family);
  CellInfoListView cells(cell_family);

  UniqueArray<Lima::Noeud> lm_nodes(mesh_nb_node);

  VariableItemReal3& nodes_coords = mesh->nodesCoordinates();

  // Sauve les noeuds
  for( Integer i=0; i<mesh_nb_node; ++i ){
    Node node = nodes[i];
    Real3 coord = nodes_coords[node];
    lm_nodes[i].set_x(Convert::toDouble(coord.x));
    lm_nodes[i].set_y(Convert::toDouble(coord.y));
    lm_nodes[i].set_z(Convert::toDouble(coord.z));
    lima.ajouter(lm_nodes[i]);
  }

  // Sauve les arêtes
  for( Integer i=0; i<mesh_nb_edge; ++i ){
    _writeItem(lima,lm_nodes,edges[i]);
  }

  // Sauve les faces
  for( Integer i=0; i<mesh_nb_face; ++i ){
    _writeItem(lima,lm_nodes,faces[i]);
  }

  // Sauve les mailles
  for( Integer i=0; i<mesh_nb_cell; ++i ){
    _writeItem(lima,lm_nodes,cells[i]);
  }

  try{

    // Sauve les groupes de noeuds
    for( ItemGroupCollection::Enumerator i(node_family->groups()); ++i; ){
      ItemGroup group = *i;
      if (group.isAllItems())
        continue;
      Lima::Nuage lm_group(group.name().localstr());
      lima.ajouter(lm_group);
      ENUMERATE_ITEM(iitem,group){
        lm_group.ajouter(lima.noeud(iitem.localId()));
      }
    }

    // Sauve les groupes d'arêtes
    for( ItemGroupCollection::Enumerator i(edge_family->groups()); ++i; ){
      ItemGroup group = *i;
      if (group.isAllItems())
        continue;
      Lima::Ligne lm_group(group.name().localstr());
      lima.ajouter(lm_group);
      ENUMERATE_ITEM(iitem,group){
        lm_group.ajouter(lima.bras(iitem.localId()));
      }
    }

    // Sauve les groupes de face
    for( ItemGroupCollection::Enumerator i(face_family->groups()); ++i; ){
      ItemGroup group = *i;
      if (group.isAllItems())
        continue;
      if (dimension==3){
        Lima::Surface lm_group(group.name().localstr());
        lima.ajouter(lm_group);
        ENUMERATE_ITEM(iitem,group){
          lm_group.ajouter(lima.polygone(iitem.localId()));
        }
      }
      else if (dimension==2){
        Lima::Ligne lm_group(group.name().localstr());
        lima.ajouter(lm_group);
        ENUMERATE_ITEM(iitem,group){
          lm_group.ajouter(lima.bras(iitem.localId()));
        }
      }
    }
 
    // Sauve les groupes de maille
    for( ItemGroupCollection::Enumerator i(cell_family->groups()); ++i; ){
      ItemGroup group = *i;
      if (group.isAllItems())
        continue;
      if (dimension==3){
        Lima::Volume lm_group(group.name().localstr());
        lima.ajouter(lm_group);
        ENUMERATE_ITEM(iitem,group){
          lm_group.ajouter(lima.polyedre(iitem.localId()));
        }
      }
      else if (dimension==2){
        Lima::Surface lm_group(group.name().localstr());
        lima.ajouter(lm_group);
        ENUMERATE_ITEM(iitem,group){
          lm_group.ajouter(lima.polygone(iitem.localId()));
        }
      }
    }
    info(4) << "Writing file '" << std_file_name << "'";

    lima.ecrire(std_file_name);
  }
  catch(const std::exception& ex){
    trace->warning() << "Exception (std::exception) in LIMA: Can not write file <" << std_file_name << ">"
                      << " Exception: " << ex.what() << '\n';
    return true;
  }
  catch(...){
    trace->warning() << "Exception (unknown) in LIMA: Can not write file <" << std_file_name << ">";
    return true;
  }

  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Créé un groupe d'entités.
 *
 * Pour assurer la reproductibilité, s'assure que les entités sont
 * triées suivant leur localid. S'assure aussi qu'il n'y a pas de doublons
 * dans la liste car lima l'autorise mais pas Arcane.
 */
void LimaUtils::
createGroup(IItemFamily* family,const String& name,Int32ArrayView local_ids)
{
  ITraceMng* tm = family->traceMng();
  if (!local_ids.empty())
    std::sort(std::begin(local_ids),std::end(local_ids));
  Integer nb_item = local_ids.size();
  Integer nb_duplicated = 0;
  // Détecte les doublons
  for( Integer i=1; i<nb_item; ++i )
    if (local_ids[i]==local_ids[i-1]){
      ++nb_duplicated;
    }
  if (nb_duplicated!=0){
    tm->warning() << "Duplicated items in group name=" << name
                  << " nb_duplicated=" << nb_duplicated;
    auto xbegin = std::begin(local_ids);
    auto xend = std::end(local_ids);
    Integer new_size = CheckedConvert::toInteger(std::unique(xbegin,xend)-xbegin);
    tm->info() << "NEW_SIZE=" << new_size << " old=" << nb_item;
    local_ids = local_ids.subView(0,new_size);
  }
  
  family->createGroup(name,local_ids,true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
