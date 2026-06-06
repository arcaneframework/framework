// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcaneTypes.h                                               (C) 2000-2026 */
/*                                                                           */
/* Definition of Arcane's general types.                                     */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ARCANETYPES_H
#define ARCANE_CORE_ARCANETYPES_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"
#include "arcane/core/datatype/DataTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// GG: Should not be here. "SerializeGlobal.h" should be included instead,
// but this is not possible as long as certain files in
// 'arcane/utils' use header files from 'arcane/core'. This is
// the case, for example, with files '*AMR*.h'.

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
 * \brief Declarations of Arcane's general types.
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
class IServiceFactory;
class IServiceMng;
class IServiceInfo;
class IServiceInstance;
class ISingletonServiceInstance;
class IModule;
class ISubDomain;
class IServiceInstance;
class ModuleBuildInfo;
class ServiceBuildInfoBase;
class ServiceBuildInfo;
class ServiceProperty;
class ServiceInstanceRef;

/*!
 * \brief Reference to the 'ISingletonServiceInstance' interface
 */
typedef Ref<ISingletonServiceInstance> SingletonServiceInstanceRef;

/*!
 * \brief Internal types of Arcane.
 *
 * These types should not be used outside of Arcane and their
 * API may be modified at any time.
 */
namespace Internal
{
  class IServiceFactory2;
  class AbstractServiceFactory;
  template <typename InterfaceType>
  class IServiceFactory2T;
  class ISingletonServiceFactory;
  class ServiceInfo;
  template <typename ServiceType>
  class ServiceAllInterfaceRegisterer;
} // namespace Internal

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class CaseOptionsMain;
class CheckpointInfo;
class CheckpointReadInfo;
class CommonVariables;
class ConnectivityItemVector;
class FileContent;
class ICheckpointReader;
class ICheckpointWriter;
class IVariableMng;
class IMeshFactoryMng;
class IMeshMng;
class IMesh;
class IMeshArea;
class IMeshCompacter;
class IMeshExchanger;
class IMeshInternal;
class IMeshBase;
class IMeshPartitionConstraint;
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
class ItemPairGroupBuilder;
class IIndexedIncrementalItemConnectivityMng;
class IIndexedIncrementalItemConnectivity;
class IMeshInitialAllocator;
class UnstructuredMeshAllocateBuildInfo;
class CartesianMeshAllocateBuildInfo;
class IConfiguration;
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
class IVariableComputeFunction;
class IMemoryAccessTrace;
class IParallelExchanger;
class IVariableSynchronizer;
class IParallelTopology;
class IParallelMngInternal;
class IAsyncParticleExchanger;
class IIOMng;
class ITimerMng;
class IThreadMng;
class ItemUniqueId;
class IItemConnectivityInfo;
class IItemConnectivity;
class IItemConnectivitySynchronizer;
class IItemConnectivityGhostPolicy;
class IItemInternalSortFunction;
class IItemConnectivityMng;
class Properties;
class IItemFamilyTopologyModifier;
class IItemFamilyPolicyMng;
class IPhysicalUnit;
class IPhysicalUnitConverter;
class IPhysicalUnitSystem;
class IDataReader;
class IDataReader2;
class IDataWriter;
class IXmlDocumentHolder;
class VariableComparer;
class VariableComparerArgs;
class VariableComparerResults;
class SubDomainBuildInfo;
class XmlNode;
class TimeLoopEntryPointInfo;
class TimeLoopSingletonServiceInfo;
class Timer;
class VariableDependInfo;
class VariableMetaData;
enum class eVariableComparerCompareMode;
enum class eVariableComparerComputeDifferenceMethod;
enum class eMeshStructure;
enum class eMeshAMRKind;
using TimeLoopEntryPointInfoCollection = Collection<TimeLoopEntryPointInfo>;
using TimeLoopSingletonServiceInfoCollection = Collection<TimeLoopSingletonServiceInfo>;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// The following classes should not be declared for SWIG because this
// file is included by SWIG and the declaration modifies how
// certain classes are generated, which leads to errors
#ifndef SWIG
class ItemVectorView;
#endif

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Mesh entity type
enum eItemKind
{
  IK_Node = 0, //!< Node mesh entity
  IK_Edge = 1, //!< Edge mesh entity
  IK_Face = 2, //!< Face mesh entity
  IK_Cell = 3, //!< Cell mesh entity
  IK_DoF = 4, //!< Degree of Freedom mesh entity
  IK_Particle = 5, //!< Particle mesh entity
  IK_Unknown = 6 //!< Unknown or uninitialized mesh entity
};

//! Number of mesh entity kinds.
static const Integer NB_ITEM_KIND = 6;

//! Entity kind name.
extern "C++" ARCANE_CORE_EXPORT const char*
itemKindName(eItemKind kind);

//! Output operator for a stream
extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& ostr, eItemKind item_kind);

//! Input operator from a stream
extern "C++" ARCANE_CORE_EXPORT std::istream&
operator>>(std::istream& istr, eItemKind& item_kind);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! \brief Number corresponding to a null entity.
  \deprecated.
*/
static const Integer NULL_ITEM_ID = static_cast<Integer>(-1);

//! Number corresponding to a null local entity
static const Integer NULL_ITEM_LOCAL_ID = static_cast<Integer>(-1);

//! Number corresponding to a null unique entity
static const Int64 NULL_ITEM_UNIQUE_ID = static_cast<Int64>(-1);

//! Number corresponding to a null subdomain
static const Integer NULL_SUB_DOMAIN_ID = static_cast<Integer>(-1);

//! Number corresponding to a null rank (for message exchange)
static const Int32 A_NULL_RANK = static_cast<Int32>(-1);

//! Number of unknown or null entity type
static const Int16 IT_NullType = 0;
//! Number of Node entity type (1 vertex, 1D, 2D, and 3D)
static const Int16 IT_Vertex = 1;
//! Number of Edge entity type (2 vertices, 1D, 2D, and 3D)
static const Int16 IT_Line2 = 2;
//! Number of Triangle entity type (3 vertices, 2D)
static const Int16 IT_Triangle3 = 3;
//! Number of Quadrilateral entity type (4 vertices, 2D)
static const Int16 IT_Quad4 = 4;
//! Number of Pentagon entity type (5 vertices, 2D)
static const Int16 IT_Pentagon5 = 5;
//! Number of Hexagon entity type (6 vertices, 2D)
static const Int16 IT_Hexagon6 = 6;
//! Number of Tetrahedron entity type (4 vertices, 3D)
static const Int16 IT_Tetraedron4 = 7;
//! Number of Pyramid entity type (5 vertices, 3D)
static const Int16 IT_Pyramid5 = 8;
//! Number of Prism entity type (6 vertices, 3D)
static const Int16 IT_Pentaedron6 = 9;
//! Number of Hexahedron entity type (8 vertices, 3D)
static const Int16 IT_Hexaedron8 = 10;
//! Number of Heptahedron entity type (prism with pentagonal base)
static const Int16 IT_Heptaedron10 = 11;
//! Number of Octahedron entity type (prism with hexagonal base)
static const Int16 IT_Octaedron12 = 12;
//! Number of HemiHexa7 entity type (hexahedron with 1 degeneracy)
static const Int16 IT_HemiHexa7 = 13;
//! Number of HemiHexa6 entity type (hexahedron with 2 non-contiguous degeneracies)
static const Int16 IT_HemiHexa6 = 14;
//! Number of HemiHexa5 entity type (hexahedrons with 3 non-contiguous degeneracies)
static const Int16 IT_HemiHexa5 = 15;
//! Number of AntiWedgeLeft6 entity type (hexahedron with 2 contiguous degeneracies)
static const Int16 IT_AntiWedgeLeft6 = 16;
//! Number of AntiWedgeRight6 entity type (hexahedron with 2 contiguous degeneracies (second form))
static const Int16 IT_AntiWedgeRight6 = 17;
//! Number of DiTetra5 entity type (hexahedron with 3 orthogonal degeneracies)
static const Int16 IT_DiTetra5 = 18;
//! Number of dual node entity type of a vertex
static const Int16 IT_DualNode = 19;
//! Number of dual edge entity type
static const Int16 IT_DualEdge = 20;
//! Number of dual face entity type
static const Int16 IT_DualFace = 21;
//! Number of dual cell entity type
static const Int16 IT_DualCell = 22;
//! Number of link entity type
static const Int16 IT_Link = 23;
//! Number of Face entity type for 1D meshes.
static const Int16 IT_FaceVertex = 24;
//! Number of Cell entity type for 1D meshes.
static const Int16 IT_CellLine2 = 25;
//! Number of dual particle entity type
static const Int16 IT_DualParticle = 26;

//! Number of Ennehedron entity type (prism with heptagonal base)
static const Int16 IT_Enneedron14 = 27;
//! Number of Decahedron entity type (prism with octagonal base)
static const Int16 IT_Decaedron16 = 28;

//! Number of 2D Heptagon entity type (heptagonal)
static const Int16 IT_Heptagon7 = 29;

//! Number of 2D Octagon entity type (octagonal)
static const Int16 IT_Octogon8 = 30;

//! Quadratic elements
//@{
//! Order 2 line
static const Int16 IT_Line3 = 31;
//! Order 2 triangle
static const Int16 IT_Triangle6 = 32;
//! Order 2 quadrangle (with 4 nodes on the faces)
static const Int16 IT_Quad8 = 33;
//! Order 2 tetrahedron
static const Int16 IT_Tetraedron10 = 34;
//! Order 2 hexahedron
static const Int16 IT_Hexaedron20 = 35;
//! Order 2 pyramid
static const Int16 IT_Pyramid13 = 36;
//! Order 2 prism
static const Int16 IT_Pentaedron15 = 37;
//@}

//! Line3 Mesh. EXPERIMENTAL!
static const Int16 IT_CellLine3 = 38;

/*!
 * \brief 2D meshes in a 3D mesh.
 * \warning These types are experimental and should not
 * be used outside of %Arcane.
 */
//@{
//! Line2 Mesh in a 3D mesh. EXPERIMENTAL!
static const Int16 IT_Cell3D_Line2 = 39;
//! Triangular Mesh with 3 nodes in a 3D mesh. EXPERIMENTAL!
static const Int16 IT_Cell3D_Triangle3 = 40;
//! Quadrangular Mesh with 4 nodes in a 3D mesh. EXPERIMENTAL!
static const Int16 IT_Cell3D_Quad4 = 41;
//! Line3 Mesh in a 3D mesh. EXPERIMENTAL!
static const Int16 IT_Cell3D_Line3 = 42;
//! Triangular Mesh with 6 nodes in a 3D mesh. EXPERIMENTAL!
static const Int16 IT_Cell3D_Triangle6 = 43;
//! Quadrangular Mesh with 8 nodes in a 3D mesh. EXPERIMENTAL!
static const Int16 IT_Cell3D_Quad8 = 44;
//! Quadrangular Mesh with 9 nodes in a 3D mesh. EXPERIMENTAL!
static const Int16 IT_Cell3D_Quad9 = 45;
//@}

//! Order 2 Quadrangle (with 4 nodes on the faces and 1 node in the center). EXPERIMENTAL!
static const Int16 IT_Quad9 = 46;
//! Order 2 Hexahedron (with 12 nodes on the edges, 6 on the faces, and one center node). EXPERIMENTAL!
static const Int16 IT_Hexaedron27 = 47;

//! Order 3 Line
static const Int16 IT_Line4 = 48;
//! Order 3 Triangle. EXPERIMENTAL!
static const Int16 IT_Triangle10 = 49;
//! Order 3 Line. EXPERIMENTAL!
static const Int16 IT_CellLine4 = 50;
//! Order 3 Line. EXPERIMENTAL!
static const Int16 IT_Cell3D_Line4 = 51;
//! Order 3 Triangle in a 3D mesh. EXPERIMENTAL!
static const Int16 IT_Cell3D_Triangle10 = 52;

//! Default number of available entity types
static const Integer NB_BASIC_ITEM_TYPE = 53;

//! First value for generic polygon types (EXPERIMENTAL)
static const Int16 IT_GenericPolygon = 200;

extern "C++" ARCANE_CORE_EXPORT eItemKind
dualItemKind(Integer type);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Phase of a temporal action.
 */
enum eTimePhase
{
  TP_Computation = 0,
  TP_Communication,
  TP_InputOutput
};
static const Integer NB_TIME_PHASE = 3;

//! Output operator on a stream
extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& ostr, eTimePhase time_phase);

//! Input operator from a stream
extern "C++" ARCANE_CORE_EXPORT std::istream&
operator>>(std::istream& istr, eTimePhase& time_phase);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

//! Direction type for a structured mesh
enum eMeshDirection
{
  //! X Direction
  MD_DirX = 0,
  //! Y Direction
  MD_DirY = 1,
  //! Z Direction
  MD_DirZ = 2,
  //! Invalid or uninitialized direction
  MD_DirInvalid = (-1)
};

//! Output operator on a stream
extern "C++" ARCANE_CORE_EXPORT std::ostream&
operator<<(std::ostream& o, eMeshDirection md);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class ISubDomain;

template <typename T> class SimplePropertyT;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

template <typename DataType>
class IScalarDataT;
template <typename DataType>
class IArrayDataT;
template <typename DataType>
class IArray2DataT;
template <typename DataType>
class IMultiArray2DataT;

template <typename DataType>
class VariableScalarT;
template <typename DataType>
class VariableArrayT;
template <typename DataType>
class Array2VariableT;

template <typename DataType>
class VariableRefScalarT;
template <typename DataType>
class VariableRefArrayT;
template <typename DataType>
class VariableRefArray2T;
// TODO: This type is no longer used. Delete by end of 2025
template <typename DataType>
class MultiArray2VariableRefT;

template <typename DataType>
class ItemVariableScalarRefT;
template <typename ItemType, class DataType>
class MeshVariableScalarRefT;
template <typename ItemType, class DataType>
class MeshVariableArrayRefT;
template <typename DataType>
class ItemPartialVariableScalarRefT;
template <typename ItemType, class DataType>
class MeshPartialVariableScalarRefT;
template <typename ItemTypeT, typename DataTypeT>
class SharedMeshVariableScalarRefT;
template <typename DataTypeT>
class SharedItemVariableScalarRefT;

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
class IItemFamilyCompactPolicy;
class IItemFamilyExchanger;
class IItemFamilySerializer;
class IItemFamilySerializeStepFactory;
class ItemFamilySerializeArgs;
class ILoadBalanceMng;
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
class ITimeLoopService;
class IVariable;
class IVariableInternal;
class VariableRef;
class Item;
class CaseOptionBuildInfo;
class CaseOptionEnum;
class CaseOptionExtended;
class CaseOptionMultiSimple;
class CaseOptionServiceImpl;
class CaseOptionSimple;
class CaseOptionMultiEnum;
class CaseOptionMultiExtended;
class CaseOptionMultiServiceImpl;
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

/*! \brief Collection of modules. */
typedef Collection<IModule*> ModuleCollection;
/*! \brief Collection of services. */
typedef Collection<IService*> ServiceCollection;
/*! \brief Collection of service instances. */
typedef Collection<ServiceInstanceRef> ServiceInstanceCollection;
/*! \brief Collection of singleton service instances. */
typedef Collection<SingletonServiceInstanceRef> SingletonServiceInstanceCollection;
/*! \brief Collection of service factories. */
typedef Collection<Internal::IServiceFactory2*> ServiceFactory2Collection;
/*! \brief Collection of service factory information. */
typedef Collection<IServiceFactoryInfo*> ServiceFactoryInfoCollection;
/*! \brief Collection of module factory information. */
typedef Collection<IModuleFactoryInfo*> ModuleFactoryInfoCollection;
/*! \brief Collection of entry points. */
typedef Collection<IEntryPoint*> EntryPointCollection;
/*! \brief Collection of time loops. */
typedef Collection<ITimeLoop*> TimeLoopCollection;
/*! \brief Collection of variables. */
typedef Collection<VariableRef*> VariableRefCollection;
/*! \brief Collection of mesh item groups. */
typedef Collection<ItemGroup> ItemGroupCollection;
/*! \brief Collection of subdomains. */
typedef Collection<ISubDomain*> SubDomainCollection;
/*! \brief Collection of sessions. */
typedef Collection<ISession*> SessionCollection;
/*! \brief Collection of dataset options. */
typedef Collection<ICaseOptions*> CaseOptionsCollection;
/*! \brief Collection of item families. */
typedef Collection<IItemFamily*> IItemFamilyCollection;
/*! \brief Collection of item families. */
typedef Collection<IItemFamily*> ItemFamilyCollection;
/*! \brief Collection of meshes. */
typedef Collection<IMesh*> IMeshCollection;
/*! \brief Collection of meshes. */
typedef Collection<IMesh*> MeshCollection;
/*! \brief Collection of tied interfaces. */
typedef Collection<ITiedInterface*> TiedInterfaceCollection;

/*! \brief Array of modules. */
typedef List<IModule*> ModuleList;
/*! \brief Array of services. */
typedef List<IService*> ServiceList;
/*! \brief Array of entry points. */
typedef List<IEntryPoint*> EntryPointList;
/*! \brief Array of time loops. */
typedef List<ITimeLoop*> TimeLoopList;
/*! \brief Array of variable references. */
typedef List<VariableRef*> VariableRefList;
/*! \brief Array of mesh item groups. */
typedef List<ItemGroup> ItemGroupList;
/*! \brief Array of subdomains. */
typedef List<ISubDomain*> SubDomainList;
/*! \brief Array of sessions. */
typedef List<ISession*> SessionList;
/*! \brief Array of dataset options. */
typedef List<ICaseOptions*> CaseOptionsList;
/*! \brief Array of tied interfaces. */
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
template <typename DataType> class IArrayDataInternalT;
template <typename DataType> class IArray2DataInternalT;

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
} // namespace Accelerator
using Accelerator::AcceleratorRuntimeInitialisationInfo;
using Accelerator::IAcceleratorMng;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// Declarations of types used for 'friend' classes.
namespace mesh
{
  class DynamicMesh;
  class ItemFamily;
  class ItemSharedInfoWithType;
  class DynamicMeshKindInfos;
  class ItemDataList;
} // namespace mesh

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
/*! \brief Collection of dataset functions. */
typedef Collection<Ref<ICaseFunction>> CaseFunctionCollection;
/*! \brief Array of dataset functions. */
typedef List<Ref<ICaseFunction>> CaseFunctionList;
} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
