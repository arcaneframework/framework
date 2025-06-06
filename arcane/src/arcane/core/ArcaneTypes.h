// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneTypes.h                                               (C) 2000-2025 */
/*                                                                           */
/* Définition des types généraux de Arcane.                                  */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ARCANETYPES_H
#define ARCANE_CORE_ARCANETYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/datatype/DataTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// GG: Ne devrait pas être ici. Il faudrait inclure "SerializeGlobal.h" à la
// place mais cela n'est pas possible tant que certains fichiers de
// 'arcane/utils' utilisent des fichiers d'en-tête de 'arcane/core'. C'est
// le cas par exemple des fichiers '*AMR*.h'.

namespace Arcane
{
class ISerializer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \file ArcaneTypes.h
 *
 * \brief Déclarations des types généraux de Arcane.
 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IBase;
class ISession;
class IApplication;
class IRessourceMng;
class Real2;
class Real3;
class Real2x2;
class Real3x3;
class IData;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IArcaneMain;
class ApplicationInfo;
class IMainFactory;
class VersionInfo;
class ITimeStats;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IService;
class IServiceInfo;
class IServiceInstance;
class ISingletonServiceInstance;
class IModule;
class ISubDomain;
class IServiceInstance;
class ServiceBuildInfoBase;
class ServiceBuildInfo;
class ServiceProperty;
class ServiceInstanceRef;

/*!
 * \brief Référence à l'interface 'ISingletonServiceInstance'
 */
typedef Ref<ISingletonServiceInstance> SingletonServiceInstanceRef;
/*!
 * \brief Types interne à Arcane.
 *
 * Ces types ne doivent pas être utilisés en dehors de Arcane et leur
 * API peut être modifiée à tout moment.
 */
namespace Internal
{
class IServiceFactory2;
class AbstractServiceFactory;
template<typename InterfaceType>
class IServiceFactory2T;
class ISingletonServiceFactory;
class ServiceInfo;
template<typename ServiceType>
class ServiceAllInterfaceRegisterer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableMng;
class IMeshFactoryMng;
class IMeshMng;
class IMesh;
class IMeshInternal;
class IMeshBase;
class IUserDataList;
class IMeshBuilder;
class MeshHandle;
class MeshHandleOrMesh;
class MeshBuildInfo;
class MeshKind;
class IPrimaryMesh;
class IMeshInitialAllocator;
class ItemFamilyCompactInfos;
class ItemFamilyItemListChangedEventArgs;
class ItemPairEnumerator;
class ItemInfoListView;
class ItemGenericInfoListView;
class IIndexedIncrementalItemConnectivityMng;
class IIndexedIncrementalItemConnectivity;
class IMeshInitialAllocator;
class UnstructuredMeshAllocateBuildInfo;
class CartesianMeshAllocateBuildInfo;
class IIncrementalItemConnectivity;
class IIncrementalItemConnectivityInternal;
class IIncrementalItemTargetConnectivity;
class IIncrementalItemSourceConnectivity;
class VariableSynchronizerEventArgs;
class IVariableSynchronizerMng;
class IParallelMng;
class IParallelMngContainer;
class IParallelMngContainerFactory;
class IParallelReplication;
class IParallelNonBlockingCollective;
class IParallelMngUtilsFactory;
class IGetVariablesValuesParallelOperation;
class ITransferValuesParallelOperation;
class IParallelExchanger;
class IVariableSynchronizer;
class IParallelTopology;
class IParallelMngInternal;
class IIOMng;
class ITimerMng;
class IThreadMng;
class ItemUniqueId;
class IItemConnectivityInfo;
class IItemConnectivity;
class IItemInternalSortFunction;
class IItemConnectivityMng;
class Properties;
class IItemFamilyTopologyModifier;
class IItemFamilyPolicyMng;
class IDataReader;
class IDataWriter;
class VariableComparer;
class VariableComparerArgs;
class VariableComparerResults;
class SubDomainBuildInfo;
enum class eVariableComparerCompareMode;
enum class eVariableComparerComputeDifferenceMethod;
enum class eMeshStructure;
enum class eMeshAMRKind;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Genre d'entité de maillage
enum eItemKind
{
  IK_Node     = 0, //!< Entité de maillage de genre noeud
  IK_Edge     = 1, //!< Entité de maillage de genre arête
  IK_Face     = 2, //!< Entité de maillage de genre face
  IK_Cell     = 3, //!< Entité de maillage de genre maille
  IK_DoF      = 4, //!< Entité de maillage de genre degre de liberte
  IK_Particle = 5, //!< Entité de maillage de genre particule
  IK_Unknown  = 6  //!< Entité de maillage de genre inconnu ou non initialisé
};

//! Nombre de genre d'entités de maillage.
static const Integer NB_ITEM_KIND = 6;

//! Nom du genre d'entité.
extern "C++" ARCANE_CORE_EXPORT const char*
itemKindName(eItemKind kind);

//! Opérateur de sortie sur un flot
extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<< (std::ostream& ostr,eItemKind item_kind);

//! Opérateur d'entrée depuis un flot
extern "C++" ARCANE_CORE_EXPORT std::istream&
operator>> (std::istream& istr,eItemKind& item_kind);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Numéro correspondant à une entité nulle.
  \deprecated.
*/
static const Integer NULL_ITEM_ID = static_cast<Integer>(-1);

//! Numéro correspondant à une entité nulle
static const Integer NULL_ITEM_LOCAL_ID = static_cast<Integer>(-1);

//! Numéro correspondant à une entité nulle
static const Int64 NULL_ITEM_UNIQUE_ID = static_cast<Int64>(-1);

//! Numéro correspondant à un sous-domaine nul
static const Integer NULL_SUB_DOMAIN_ID = static_cast<Integer>(-1);

//! Numéro correspondant à un rang nul (pour l'échange de message)
static const Int32 A_NULL_RANK = static_cast<Int32>(-1);

//! Numéro du type d'entité inconnu ou null
static const Int16 IT_NullType = 0;
//! Numéro du type d'entité Noeud (1 sommet 1D, 2D et 3D)
static const Int16 IT_Vertex = 1;
//! Numéro du type d'entité Arête (2 sommets, 1D, 2D et 3D)
static const Int16 IT_Line2 = 2;
//! Numéro du type d'entité Triangle (3 sommets, 2D)
static const Int16 IT_Triangle3 = 3;
//! Numéro du type d'entité Quadrilatère (4 sommets, 2D)
static const Int16 IT_Quad4 = 4;
//! Numéro du type d'entité Pentagone (5 sommets, 2D)
static const Int16 IT_Pentagon5 = 5;
//! Numéro du type d'entité Hexagone (6 sommets, 2D)
static const Int16 IT_Hexagon6 = 6;
//! Numéro du type d'entité Tetraèdre (4 sommets, 3D)
static const Int16 IT_Tetraedron4 = 7;
//! Numéro du type d'entité Pyramide (5 sommets, 3D)
static const Int16 IT_Pyramid5 = 8;
//! Numéro du type d'entité Prisme (6 sommets, 3D)
static const Int16 IT_Pentaedron6 = 9;
//! Numéro du type d'entité Hexaèdre (8 sommets, 3D)
static const Int16 IT_Hexaedron8 = 10;
//! Numéro du type d'entité Heptaèdre (prisme à base pentagonale)
static const Int16 IT_Heptaedron10 = 11;
//! Numéro du type d'entité Octaèdre (prisme à base hexagonale)
static const Int16 IT_Octaedron12 = 12;
//! Numéro du type d'entité HemiHexa7 (héxahèdre à 1 dégénérescence)
static const Int16 IT_HemiHexa7 = 13;
//! Numéro du type d'entité HemiHexa6 (héxahèdre à 2 dégénérescences non contigues)
static const Int16 IT_HemiHexa6 = 14;
//! Numéro du type d'entité HemiHexa5 (héxahèdres à 3 dégénérescences non contigues)
static const Int16 IT_HemiHexa5 = 15;
//! Numéro du type d'entité AntiWedgeLeft6 (héxahèdre à 2 dégénérescences contigues)
static const Int16 IT_AntiWedgeLeft6 = 16;
//! Numéro du type d'entité AntiWedgeRight6 (héxahèdre à 2 dégénérescences contigues (seconde forme))
static const Int16 IT_AntiWedgeRight6 = 17;
//! Numéro du type d'entité DiTetra5 (héxahèdre à 3 dégénérescences orthogonales)
static const Int16 IT_DiTetra5 = 18;
//! Numero du type d'entite noeud dual d'un sommet
static const Int16 IT_DualNode = 19;
//! Numero du type d'entite noeud dual d'une arête
static const Int16 IT_DualEdge = 20;
//! Numero du type d'entite noeud dual d'une face
static const Int16 IT_DualFace = 21;
//! Numero du type d'entite noeud dual d'une cellule
static const Int16 IT_DualCell = 22;
//! Numéro du type d'entité liaison
static const Int16 IT_Link = 23;
//! Numéro du type d'entité Face pour les maillages 1D.
static const Int16 IT_FaceVertex = 24;
//! Numéro du type d'entité Cell pour les maillages 1D.
static const Int16 IT_CellLine2 = 25;
//! Numero du type d'entite noeud dual d'une particule
static const Int16 IT_DualParticle = 26;

//! Numéro du type d'entité Enneèdre (prisme à base heptagonale)
static const Int16 IT_Enneedron14 = 27;
//! Numéro du type d'entité Decaèdre (prisme à base Octogonale)
static const Int16 IT_Decaedron16 = 28;

//! Numéro du type d'entité Heptagon 2D (heptagonale)
static const Int16 IT_Heptagon7 = 29;

//! Numéro du type d'entité Octogon 2D (Octogonale)
static const Int16 IT_Octogon8 = 30;

//! Éléments quadratiques
//@{
//! Ligne d'ordre 2
static const Int16 IT_Line3 = 31;
//! Triangle d'ordre 2
static const Int16 IT_Triangle6 = 32;
//! Quadrangle d'ordre 2 (avec 4 noeuds sur les faces)
static const Int16 IT_Quad8 = 33;
//! Tétraèdre d'ordre 2
static const Int16 IT_Tetraedron10 = 34;
//! Hexaèdre d'ordre 2
static const Int16 IT_Hexaedron20 = 35;
//@}

/*!
 * \brief Mailles 2D dans un maillage 3D.
 * \warning Ces types sont expérimentaux et ne doivent
 * pas être utilisés en dehors de %Arcane.
 */
//@{
//! Maille Line2 dans un maillage 3D. EXPERIMENTAL !
static const Int16 IT_Cell3D_Line2 = 36;
//! Maille Triangulaire à 3 noeuds dans un maillage 3D. EXPERIMENTAL !
static const Int16 IT_Cell3D_Triangle3 = 37;
//! Maille Quadrangulaire à 5 noeuds dans un maillage 3D. EXPERIMENTAL !
static const Int16 IT_Cell3D_Quad4 = 38;
//@}

//! Nombre de types d'entités disponible par défaut
static const Integer NB_BASIC_ITEM_TYPE = 39;

extern "C++" ARCANE_CORE_EXPORT eItemKind
dualItemKind(Integer type);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Phase d'une action temporelle.
 */
enum eTimePhase
{
  TP_Computation = 0,
  TP_Communication,
  TP_InputOutput
};
static const Integer NB_TIME_PHASE = 3;

//! Opérateur de sortie sur un flot
extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<< (std::ostream& ostr,eTimePhase time_phase);

//! Opérateur d'entrée depuis un flot
extern "C++" ARCANE_CORE_EXPORT std::istream&
operator>> (std::istream& istr,eTimePhase& time_phase);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Type de la direction pour un maillage structuré
enum eMeshDirection
{
  //! Direction X
  MD_DirX = 0,
  //! Direction Y
  MD_DirY = 1,
  //! Direction Z
  MD_DirZ = 2,
  //! Direction invalide ou non initialisée
  MD_DirInvalid = (-1)
};

//! Opérateur de sortie sur un flot
extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o,eMeshDirection md);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;

template<typename T> class SimplePropertyT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template<typename DataType>
class IScalarDataT;
template<typename DataType>
class IArrayDataT;
template<typename DataType>
class IArray2DataT;
template<typename DataType>
class IMultiArray2DataT;

template<typename DataType>
class VariableScalarT;
template<typename DataType>
class VariableArrayT;
template<typename DataType>
class Array2VariableT;

template<typename DataType>
class VariableRefScalarT;
template<typename DataType>
class VariableRefArrayT;
template<typename DataType>
class VariableRefArray2T;
// TODO: Ce type n'est plus utilisé. A supprimer fin 2025
template<typename DataType>
class MultiArray2VariableRefT;

template<typename DataType>
class ItemVariableScalarRefT;
template<typename ItemType,class DataType>
class MeshVariableScalarRefT;
template<typename ItemType,class DataType>
class MeshVariableArrayRefT;
template<typename DataType>
class ItemPartialVariableScalarRefT;
template<typename ItemType,class DataType>
class MeshPartialVariableScalarRefT;
template<typename ItemTypeT, typename DataTypeT>
class SharedMeshVariableScalarRefT;
template<typename DataTypeT>
class SharedItemVariableScalarRefT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class DataViewSetter;
template <typename DataType>
class DataViewGetter;
template <typename DataType>
class DataViewGetterSetter;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename ItemType, typename DataType, typename Extents>
class MeshMDVariableRefBaseT;
template <typename ItemType, typename DataType, typename Extents>
class MeshMDVariableRefT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CollectionBase;
class IItemFamily;
class IItemFamilyInternal;
class IMesh;
class IMeshPartitioner;
class IMeshPartitionerBase;
class IModule;
class IService;
class IServiceFactoryInfo;
class IModuleFactoryInfo;
class IServiceInstance;
class IEntryPoint;
class ITimeLoop;
class IVariable;
class IVariableInternal;
class VariableRef;
class Item;
class CaseOptionBuildInfo;
class ICaseOptions;
class ICaseFunction;
class ICaseDocument;
class ICaseDocumentFragment;
class ICaseMng;
class ICaseOptionList;
class IPhysicalUnitSystem;
class ItemGroup;
class ITiedInterface;
class VariableCollection;
typedef VariableCollection VariableList;

/*! \brief Collection de modules. */
typedef Collection<IModule*> ModuleCollection;
/*! \brief Collection de services. */
typedef Collection<IService*> ServiceCollection;
/*! \brief Collection d'instances de services. */
typedef Collection<ServiceInstanceRef> ServiceInstanceCollection;
/*! \brief Collection d'instances de services singletons. */
typedef Collection<SingletonServiceInstanceRef> SingletonServiceInstanceCollection;
/*! \brief Collection de fabriques de service. */
typedef Collection<Internal::IServiceFactory2*> ServiceFactory2Collection;
/*! \brief Collection d'informations sur les fabriques de service. */
typedef Collection<IServiceFactoryInfo*> ServiceFactoryInfoCollection;
/*! \brief Collection d'informations sur les fabriques de module. */
typedef Collection<IModuleFactoryInfo*> ModuleFactoryInfoCollection;
/*! \brief Collection de points d'entrées. */
typedef Collection<IEntryPoint*> EntryPointCollection;
/*! \brief Collection de boucles en temps. */
typedef Collection<ITimeLoop*> TimeLoopCollection;
/*! \brief Collection de variables. */
typedef Collection<VariableRef*> VariableRefCollection;
/*! \brief Collection de groupes d'éléments du maillage. */
typedef Collection<ItemGroup> ItemGroupCollection;
/*! \brief Collection de sous-domaines. */
typedef Collection<ISubDomain*> SubDomainCollection;
/*! \brief Collection de sessions. */
typedef Collection<ISession*> SessionCollection;
/*! \brief Collection d'options du jeu de données. */
typedef Collection<ICaseOptions*> CaseOptionsCollection;
/*! \brief Collection de familles d'entités. */
typedef Collection<IItemFamily*> IItemFamilyCollection;
/*! \brief Collection de familles d'entités. */
typedef Collection<IItemFamily*> ItemFamilyCollection;
/*! \brief Collection de maillages. */
typedef Collection<IMesh*> IMeshCollection;
/*! \brief Collection de maillages. */
typedef Collection<IMesh*> MeshCollection;
/*! \brief Collection d'interfaces liées. */
typedef Collection<ITiedInterface*> TiedInterfaceCollection;

/*! \brief Tableau de modules. */
typedef List<IModule*> ModuleList;
/*! \brief Tableau de services. */
typedef List<IService*> ServiceList;
/*! \brief Tableau de points d'entrées. */
typedef List<IEntryPoint*> EntryPointList;
/*! \brief Tableau de boucles en temps. */
typedef List<ITimeLoop*> TimeLoopList;
/*! \brief Tableau de références de variables. */
typedef List<VariableRef*> VariableRefList;
/*! \brief Tableau de groupes d'éléments du maillage. */
typedef List<ItemGroup> ItemGroupList;
/*! \brief Tableau de sous-domaines. */
typedef List<ISubDomain*> SubDomainList;
/*! \brief Tableau de sessions. */
typedef List<ISession*> SessionList;
/*! \brief Tableau d'options du jeu de données. */
typedef List<ICaseOptions*> CaseOptionsList;
/*! \brief Tableau d'interfaces liées. */
typedef List<ITiedInterface*> TiedInterfaceList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IDataVisitor;
class IScalarDataVisitor;
class IArrayDataVisitor;
class IArray2DataVisitor;
class IMultiArray2DataVisitor;
class IDataOperation;
class ISerializedData;
class IHashAlgorithm;
class DataAllocationInfo;
class DataStorageTypeInfo;
class ISerializedData;
class IDataFactoryMng;
class IDataStorageFactory;
class DataStorageBuildInfo;
class CaseDatasetSource;
class IDataInternal;
template<typename DataType> class IArrayDataInternalT;
template<typename DataType> class IArray2DataInternalT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IVariableRef;
class IVariableFactory;
class VariableTypeInfo;
class VariableRef;
class VariableBuildInfo;
class NullVariableBuildInfo;
class VariableFactoryRegisterer;
class VariableInfo;
typedef VariableRef* (*VariableFactoryVariableRefCreateFunc)(const VariableBuildInfo& vb);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Accelerator
{
class IAcceleratorMng;
class AcceleratorRuntimeInitialisationInfo;
}
using Accelerator::IAcceleratorMng;
using Accelerator::AcceleratorRuntimeInitialisationInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
// Déclarations de types utilisés pour les classes 'friend'.
namespace mesh
{
class DynamicMesh;
class ItemFamily;
class ItemSharedInfoWithType;
class DynamicMeshKindInfos;
class ItemDataList;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::IData)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::ISerializedData)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::ICaseFunction)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::ICaseOptions)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::ICaseMng)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::ICaseOptionList)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::IIncrementalItemSourceConnectivity)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::IIncrementalItemTargetConnectivity)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::IParallelMng)
ARCCORE_DECLARE_REFERENCE_COUNTED_CLASS(Arcane::IParallelMngContainer)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*! \brief Collection de fonctions du jeu de données. */
typedef Collection<Ref<ICaseFunction>> CaseFunctionCollection;
/*! \brief Tableau de fonctions du jeu de données. */
typedef List<Ref<ICaseFunction>> CaseFunctionList;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
