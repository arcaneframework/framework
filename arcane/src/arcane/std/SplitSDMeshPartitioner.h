// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* SplitSDMeshPartitioner.h                                    (C) 2000-2024 */
/*                                                                           */
/* Mesh partitioner replicating the functionality of SplitSD.                */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_FILES_SPLITSDMESHPARTITIONER_H
#define ARCANE_FILES_SPLITSDMESHPARTITIONER_H

// #define DEBUG_PARTITIONER

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#include "arcane/std/MeshPartitionerBase.h"

#include "arcane/std/SplitSDMeshPartitioner_axl.h"

#define MPICH_SKIP_MPICXX
#define OMPI_SKIP_MPICXX
#include <mpi.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
#define TAG_MAILLAGEMAITRE 1

#define CHECK_MPI_PACK_ERR(ier) \
  if (ier != MPI_SUCCESS) { \
    switch (ier) { \
    case MPI_ERR_COMM: \
      InfoProc->m_service->pfatal() << "error on MPI_Pack of type MPI_ERR_COMM"; \
      break; \
    case MPI_ERR_TYPE: \
      InfoProc->m_service->pfatal() << "error on MPI_Pack of type MPI_ERR_TYPE"; \
      break; \
    case MPI_ERR_COUNT: \
      InfoProc->m_service->pfatal() << "error on MPI_Pack of type MPI_ERR_COUNT"; \
      break; \
    case MPI_ERR_ARG: \
      InfoProc->m_service->pfatal() << "error on MPI_Pack of type MPI_ERR_ARG"; \
      break; \
    default: \
      InfoProc->m_service->pfatal() << "error on MPI_Pack of unknown type !"; \
    } \
  }

#ifndef CHECK_IF_NOT_NULL
/**
  Macros that test whether the pointer argument is not null
  and calls pfatal() if this pointer turns out to be null.
  This macro should be used after any call to a function using memory
  (malloc, calloc, realloc ...).

  @memo   Tests if a pointer is not null.

  @param  Ptr a pointer

  @author  Eric Brière de l'Isle, ONERA, DRIS/SRL

*/
#define CHECK_IF_NOT_NULL(Ptr) \
  if (Ptr == NULL) { \
    InfoProc->m_service->pfatal() << "Pointer is null, (Perhaps we lack memory !!! )"; \
  }
#endif

#ifndef MAX
#define MAX(a, b) ((a) < (b) ? (b) : (a))
#endif
#ifndef MIN
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#endif
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**     Structure describing a processor.
        @name  StrucInfoProc
*/
typedef struct T_InfoProc
{
  /*@{*/
  /** Processor number, this number is between 0 and the number of processors minus one.*/
  int me;
  /** Total number of processors,
      this is the maximum number of processors that this application
      can use during an execution.*/
  int nbSubDomain;
  //   /** Master processor number,
  //       this is the processor on which aggregation,
  //       insertion and repartitioning are performed.*/
  //   int      NoMaitre;
  /** MPI work group
      (defining a new group allows limiting the scope
      of the messages used).*/
  MPI_Comm Split_Comm;

  /// For trace management (messages, errors, fatal ...)
  Arcane::AbstractService* m_service;

  /*@}*/
} StrucInfoProc;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**
    Structure for storing local information, to avoid recalculating it multiple times.
    It essentially consists of nodes on the interfaces with different neighboring domains.
    EB, 2007. Simplification of StructureBlocEtendu
*/

typedef struct T_INTERFACE
{
  /**     Element allowing the description of an interface with a single neighboring block
	  @name  StructureInterface
  */

  /** Neighboring domain number starts at 0	*/
  int NoDomVois;
  /** List of interface nodes		*/
  Arcane::UniqueArray<Arcane::Node> ListeNoeuds;

} StructureInterface;

typedef struct T_BLOC_ETENDU
{
  /**     Element allowing the description of an extended block and its environment
	  @name  StructureBlocEtendu
  */

  /**      Number of elements */
  Integer NbElements;

  /**      Total weight of the domain */
  double PoidsDom;

  /**      Number of interfaces  */
  Integer NbIntf;

  /**      Array of interface structures, one for each neighbor of the block in question  */
  StructureInterface* Intf;

} StructureBlocEtendu;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/**     Element allowing the description of an interface between two blocks.
        @name  StrucListeVoisMail
        @see   StrucListeDomMail
*/
typedef struct T_ListeVoisMail
{
  /*@{*/
  /** Neighboring domain number (from 0 to the number of processors minus one) */
  int NoDomVois;
  /** Number of nodes in the common interface */
  int NbNoeudsInterface;
  /** Quantity to transfer through this interface */
  double Delta;
  /** Weight for the interface */
  //   int                 PoidsInterface;  no meaning here
  /*@}*/
} StrucListeVoisMail;

/**     Element to describe a block in the global mesh description.
        @name  StrucListeDomMail
        @see  StrucMaillage
*/
typedef struct T_ListeDomMail
{
  /*@{*/
  /** Number of elements in the domain */
  int NbElements;
  /** Weight for the domain */
  double Poids;
  /** Number of neighboring blocks */
  int NbVoisins;
  /** List of descriptors for the neighbors */
  StrucListeVoisMail* ListeVoisins;
  /*@}*/
} StrucListeDomMail;

/**     Structure allowing the global description of the mesh.\\

  {\em Note:} the storage of unused global numbers
  is performed on the master processor,
  the master may also change, which is transmitted to the new
  master during the master mesh update (\Ref{MAJMaillageMaitre}).

  @name  StrucMaillage
  @see   main, MAJMaillageMaitre
*/
typedef struct T_Maillage
{
  /*@{*/
  /** Number of elements in the global mesh */
  int NbElements;
  /** Number of elements in the global mesh */
  double Poids;

  /** Number of non-empty domains */
  int NbDomainesPleins;
  /** Total number of domains, this is the number of processors assigned to this session */
  int NbDomainesMax;
  /** List (of size NbDomainesMax) of block descriptors */
  StrucListeDomMail* ListeDomaines;

  /** Number of processors whose subdomain is empty */
  int NbProcsVides;
  //   /** List of processors without a subdomain */
  //   int*                ListeNoProcsVides;

  /*@}*/
} StrucMaillage;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Mesh partitioner inspired by the SplitSD library, developed
 *  initially at ONERA for Dassault Aviation.
 */
class SplitSDMeshPartitioner
: public ArcaneSplitSDMeshPartitionerObject
//: public MeshPartitionerBase
{
 public:

  SplitSDMeshPartitioner(const ServiceBuildInfo& sbi);

 public:

  virtual void build() {}

 public:

  virtual void partitionMesh(bool initial_partition);
  virtual void partitionMesh(bool initial_partition, Int32 nb_part)
  {
    ARCANE_UNUSED(initial_partition);
    ARCANE_UNUSED(nb_part);
    throw NotImplementedException(A_FUNCINFO);
  }

 private:

  /// initialization of structures
  void init(bool initial_partition, StrucInfoProc*& InfoProc, StructureBlocEtendu*& Domaine, StrucMaillage*& Maillage);

  /// initialization of weights (m_cells_weight => m_poids_aux_mailles)
  void initPoids(bool initial_partition);

  /// memory freeing of structures
  void fin(StrucInfoProc*& InfoProc, StructureBlocEtendu*& Domaine, StrucMaillage*& Maillage);

  /// update of the local structure (nodes on interfaces with neighboring subdomains)
  void MAJDomaine(StructureBlocEtendu* Domaine);

  /// update of the structure on processor 0
  void MAJMaillageMaitre(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage);

  /// consistency check (reciprocity) of the interfaces
  void verifMaillageMaitre(StrucMaillage* Maillage);

  /// We use a front-tracking method to go from an overloaded node to other nodes by memorizing the path to update the Deltas on the interfaces.
  void MAJDeltaGlobal(StrucInfoProc* InfoProc, StrucMaillage* Maillage, double tolerance);
  /// function to shift the Delta associated with a searched interface for a pair, domain, and neighbor number, specified
  void MAJDelta(double don, int iDOmTmpPrec, int iDomTmp, StrucListeDomMail* ListeDomaines);

  /// calculates a deltaMin based on local transfers
  double CalculDeltaMin(StrucMaillage* Maillage, double deltaMin, int iterEquilibrage, int NbMaxIterEquil);

  /// iterative phase to balance the load
  void Equilibrage(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage);

  /// transfer phase between 2 domains, MAJ of domains
  void Equil2Dom(int* MasqueDesNoeuds, int* MasqueDesElements, int marqueVu, int marqueNonVu,
                 StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage,
                 int indDomCharge, int indDomVois, double Delta);

  /// selection of elements in a domain for balancing between 2 domains, by performing a front-tracking from the interface
  void SelectElements(int* MasqueDesNoeuds, int* MasqueDesElements, int marqueVu, int marqueNonVu,
                      StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine,
                      double Delta, int indDomVois, Arcane::Array<Arcane::Cell>& ListeElements);

  /// limited front-tracking following the Delta (cumulative weight of elements taken in the fronts), returns an integer in case of blockage
  int ParcoursFrontalDelta(int* MasqueDesNoeuds, int* MasqueDesElements,
                           int marqueVu, int marqueNonVu,
                           double Delta,
                           int* pNbFronts, int NbFrontsMax,
                           Arcane::Array<Arcane::Node>& FrontsNoeuds, int* IndFrontsNoeuds,
                           Arcane::Array<Arcane::Cell>& FrontsElements, int* IndFrontsElements);

  /** smoothing of the last front obtained by front-tracking,
      in order to include in this front the elements whose all nodes
      are already taken in previous fronts
  */
  void LissageDuFront(int* MasqueDesNoeuds, int* MasqueDesElements,
                      int marqueVu, int marqueNonVu,
                      int NbFronts,
                      Arcane::Array<Arcane::Node>& FrontsNoeuds, int* IndFrontsNoeuds,
                      Arcane::Array<Arcane::Cell>& FrontsElements, int* IndFrontsElements);

  /// makes the domain connected when parts are too small and not connected
  void ConnexifieDomaine(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, StrucMaillage* Maillage, double tolConnexite);

  /// searches for the neighboring domain having the max number of shared Faces with the set of meshes
  int getDomVoisMaxFace(Arcane::Array<Arcane::Cell>& ListeElements, int me);

  /// creation of an array that serves as a mask on the LocalId of nodes
  int* GetMasqueDesNoeuds(StrucInfoProc* InfoProc);
  /// creation of an array that serves as a mask on the LocalId of elements
  int* GetMasqueDesElements(StrucInfoProc* InfoProc);

  void LibereInfoProc(StrucInfoProc*& InfoProc);
  void LibereDomaine(StructureBlocEtendu*& Domaine);
  void LibereMaillage(StrucMaillage*& Maillage);

  void AfficheDomaine(int NbDom, StructureBlocEtendu* Domaine);
  void AfficheMaillage(StrucMaillage* Maillage);
  void AfficheListeDomaines(StrucListeDomMail* ListeDomaines, int NbDomaines);
  void AfficheEquilMaillage(StrucMaillage* Maillage);

  void* RecoitMessage(StrucInfoProc* InfoProc, int FromProc, int Tag, int* pTailleTMP);
  void EnvoieMessage(StrucInfoProc* InfoProc, int ToProc, int Tag, void* TabTMP, int TailleTMP);
  //   MPI_Request*  EnvoieIMessage(StrucInfoProc* InfoProc, int ToProc, int Tag, void* TabTMP, int TailleTMP);
  void* DiffuseMessage(StrucInfoProc* InfoProc, int FromProc, void* TabTMP, int TailleTMP);

  // size for transfer without nodes, only the size of ListeNoeuds
  int TailleDom(StructureBlocEtendu* Domaine);
  // the data is put into an array
  void PackDom(StrucInfoProc* InfoProc, StructureBlocEtendu* Domaine, void* TabTMP, int TailleTMP, MPI_Comm comm);
  // the data is extracted from an array
  void UnpackDom(void* TabTMP, int TailleTMP, MPI_Comm comm, StrucListeDomMail* DomMail);

  // size for domain transfer for Balancing, and for Delta
  int TailleEquil();
  // the data is put into an array
  void PackEquil(StrucInfoProc* InfoProc, int indDomCharge, int indDomVois, double Delta, void* TabTMP, int TailleTMP, MPI_Comm comm);
  // the data is extracted from an array
  void UnpackEquil(void* TabTMP, int TailleTMP, MPI_Comm comm, int* indDomCharge, int* indDomVois, double* Delta);

 private:

  VariableCellReal m_poids_aux_mailles; // weight calculated once per rebalancing call and follows the cells
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
