// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* IParallelMng.h                                              (C) 2000-2025 */
/*                                                                           */
/* Interface du gestionnaire du parallélisme sur un sous-domaine.            */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_IPARALLELMNG_H
#define ARCANE_CORE_IPARALLELMNG_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/UtilsTypes.h"

#include "arcane/core/Parallel.h"
#include "arcane/core/VariableTypedef.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Parallel
 * \brief Interface du gestionnaire de parallélisme pour un sous-domaine.
 *
 * Ce gestionnaire propose une interface pour accéder à
 * l'ensemble des fonctionnalités liées au parallélisme.
 *
 * Il existe plusieurs implémentations possibles:
 * - mode séquentiel.
 * - mode parallèle via MPI
 * - mode parallèle via les threads.
 * - mode parallèle mixte MPI/threads.
 * Le choix de l'implémentation se fait lors du lancement de l'application.
 *
 * Lorsqu'une opération est collective, tous les gestionnaires associés doivent
 * participer.
 *
 * Il est possible de créer à partir d'une instance un autre gestionnaire
 * contenant un sous-ensemble de rang via createSubParallelMng().
 *
 */
class ARCANE_CORE_EXPORT IParallelMng
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();
  // Classe pour accéder à _internalUtilsFactory()
  friend class ParallelMngUtilsAccessor;

 public:

  // NOTE: Laisse temporairement ce destructeur publique tant que
  // les méthodes createParallelMng() existent pour des raisons de
  // compatibilité avec l'existant
  virtual ~IParallelMng() = default; //!< Libère les ressources.

 public:

  typedef Parallel::Request Request;
  using PointToPointMessageInfo = Parallel::PointToPointMessageInfo;
  using MessageId = Parallel::MessageId;
  using MessageSourceInfo = Parallel::MessageSourceInfo;
  typedef Parallel::eReduceType eReduceType;
  typedef Parallel::IStat IStat;
  
 public:

  //! Construit l'instance.
  virtual void build() =0;

 public:

  /*!
   * \brief Retourne \a true si l'exécution est parallèle.
   *
   * L'exécution est parallèle si l'instance implémente
   * un mécanisme d'échange de message tel que MPI.
   */
  virtual bool isParallel() const =0;

 private:

  // NOTE: on laisse temporairement ces deux méthodes pour garder la compatibilité binaire.

  //! Numéro du sous-domaine associé à ce gestionnaire.
  virtual ARCANE_DEPRECATED Integer subDomainId() const final { return commRank(); }

  //! Nombre total de sous-domaines.
  virtual ARCANE_DEPRECATED Integer nbSubDomain() const final { return commSize(); }

 public:

  //! Rang de cette instance dans le communicateur
  virtual Int32 commRank() const =0;

  //! Nombre d'instance dans le communicateur
  virtual Int32 commSize() const =0;

  /*!
   * \brief Adresse du communicateur MPI associé à ce gestionnaire.
   *
   * Le communicateur n'est valide que si on utilise MPI. Sinon, l'adresse
   * retournée est 0. La valeur retournée est de type (MPI_Comm*).
   */
  virtual void* getMPICommunicator() =0;

  /*!
   * \brief Adresse du communicateur MPI associé à ce gestionnaire.
   *
   * \deprecated Utiliser getMPICommunicator() à la place.
   */
  virtual ARCANE_DEPRECATED_120 void* mpiCommunicator();

  /*!
   * \brief Communicateur MPI associé à ce gestionnaire
   *
   * Le communicateur n'est valide que si on utilise MPI. Il est possible
   * de tester la validité en appelant la méthode Communicator::isValid().
   * S'il est valide, il est possible de récupérer sa valeur via un cast:
   * \code
   * IParallelMng* pm = ...;
   * MPI_Comm c = static_cast<MPI_Comm>(pm->communicator());
   * \endcode
   */
  virtual Parallel::Communicator communicator() const =0;

  /**
   * \brief Communicateur MPI issus du communicateur \a communicator()
   * réunissant tous les processus du noeud de calcul.
   *
   * Le communicateur n'est valide que si on utilise MPI. Il est possible
   * de tester la validité en appelant la méthode Communicator::isValid().
   * S'il est valide, il est possible de récupérer sa valeur via un cast:
   * \code
   * IParallelMng* pm = ...;
   * MPI_Comm c = static_cast<MPI_Comm>(pm->communicator());
   * \endcode
   */
  virtual Parallel::Communicator machineCommunicator() const =0;

  /*!
   * \brief Indique si l'implémentation utilise les threads.
   *
   * L'implémentation utilise les threads soit en mode
   * thread pure, soit en mode mixte MPI/thread.
   */
  virtual bool isThreadImplementation() const =0;

  /*!
   * \brief Indique si l'implémentation utilise le mode hybride.
   *
   * L'implémentation utilise le mode mixte MPI/thread.
   */
  virtual bool isHybridImplementation() const =0;

  //! Positionne le gestionnaire de statistiques
  virtual void setTimeStats(ITimeStats* time_stats) =0;

  //! Gestionnaire de statistiques associé (peut être nul)
  virtual ITimeStats* timeStats() const =0;

  //! Gestionnaire de traces
  virtual ITraceMng* traceMng() const =0;

  //! Gestionnaire de threads
  virtual IThreadMng* threadMng() const =0;

  //! Gestionnaire de timers
  virtual ITimerMng* timerMng() const =0;

  //! Gestionnaire des entrées/sorties
  virtual IIOMng* ioMng() const =0;

  //! Gestionnaire de parallélisme sur l'ensemble des ressources allouées
  virtual IParallelMng* worldParallelMng() const =0;

  //! Initialise le gestionnaire du parallélisme
  virtual void initialize() =0;

  //! Collecteur Arccore des statistiques temporelles (peut être nul)
  virtual ITimeMetricCollector* timeMetricCollector() const =0;

 public:


 public:
    
  //! \a true si l'instance est un gestionnaire maître des entrées/sorties.
  virtual bool isMasterIO() const =0;

  /*!
    \brief Rang de l'instance gérant les entrées/sorties (pour laquelle isMasterIO() est vrai)
    *
    * Dans l'implémentation actuelle, il s'agit toujours du processeur de rang 0.
    */
  virtual Integer masterIORank() const =0;

  //! @name allGather
  //@{
  /*!
   * \brief Effectue un regroupement sur tous les processeurs.
   * Il s'agit d'une opération collective. Le tableau \a send_buf
   * doit avoir la même taille, notée \a n, pour tous les processeurs et
   * le tableau \a recv_buf doit avoir une taille égale au nombre
   * de processeurs multiplié par \a n.
   */
  virtual void allGather(ConstArrayView<char> send_buf,ArrayView<char> recv_buf) =0;
  virtual void allGather(ConstArrayView<unsigned char> send_buf,ArrayView<unsigned char> recv_buf) =0;
  virtual void allGather(ConstArrayView<signed char> send_buf,ArrayView<signed char> recv_buf) =0;
  virtual void allGather(ConstArrayView<short> send_buf,ArrayView<short> recv_buf) =0;
  virtual void allGather(ConstArrayView<unsigned short> send_buf,ArrayView<unsigned short> recv_buf) =0;
  virtual void allGather(ConstArrayView<int> send_buf,ArrayView<int> recv_buf) =0;
  virtual void allGather(ConstArrayView<unsigned int> send_buf,ArrayView<unsigned int> recv_buf) =0;
  virtual void allGather(ConstArrayView<long> send_buf,ArrayView<long> recv_buf) =0;
  virtual void allGather(ConstArrayView<unsigned long> send_buf,ArrayView<unsigned long> recv_buf) =0;
  virtual void allGather(ConstArrayView<long long> send_buf,ArrayView<long long> recv_buf) =0;
  virtual void allGather(ConstArrayView<unsigned long long> send_buf,ArrayView<unsigned long long> recv_buf) =0;
  virtual void allGather(ConstArrayView<float> send_buf,ArrayView<float> recv_buf) =0;
  virtual void allGather(ConstArrayView<double> send_buf,ArrayView<double> recv_buf) =0;
  virtual void allGather(ConstArrayView<long double> send_buf,ArrayView<long double> recv_buf) =0;
  virtual void allGather(ConstArrayView<APReal> send_buf,ArrayView<APReal> recv_buf) =0;
  virtual void allGather(ConstArrayView<Real2> send_buf,ArrayView<Real2> recv_buf) =0;
  virtual void allGather(ConstArrayView<Real3> send_buf,ArrayView<Real3> recv_buf) =0;
  virtual void allGather(ConstArrayView<Real2x2> send_buf,ArrayView<Real2x2> recv_buf) =0;
  virtual void allGather(ConstArrayView<Real3x3> send_buf,ArrayView<Real3x3> recv_buf) =0;
  virtual void allGather(ConstArrayView<HPReal> send_buf,ArrayView<HPReal> recv_buf) =0;
  virtual void allGather(ISerializer* send_serializer,ISerializer* recv_serializer) =0;
  //@}

  //! @name gather
  //@{
  /*!
   * \brief Effectue un regroupement sur un processeurs.
   * Il s'agit d'une opération collective. Le tableau \a send_buf
   * doit avoir la même taille, notée \a n, pour tous les processeurs et
   * le tableau \a recv_buf pour le processeur \a rank doit avoir une taille égale au nombre
   * de processeurs multiplié par \a n. Ce tableau \a recv_buf est inutilisé pour
   * les autres rangs que \a rank.
   */
  virtual void gather(ConstArrayView<char> send_buf,ArrayView<char> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<unsigned char> send_buf,ArrayView<unsigned char> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<signed char> send_buf,ArrayView<signed char> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<short> send_buf,ArrayView<short> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<unsigned short> send_buf,ArrayView<unsigned short> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<int> send_buf,ArrayView<int> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<unsigned int> send_buf,ArrayView<unsigned int> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<long> send_buf,ArrayView<long> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<unsigned long> send_buf,ArrayView<unsigned long> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<long long> send_buf,ArrayView<long long> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<unsigned long long> send_buf,ArrayView<unsigned long long> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<float> send_buf,ArrayView<float> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<double> send_buf,ArrayView<double> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<long double> send_buf,ArrayView<long double> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<APReal> send_buf,ArrayView<APReal> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<Real2> send_buf,ArrayView<Real2> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<Real3> send_buf,ArrayView<Real3> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<Real2x2> send_buf,ArrayView<Real2x2> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<Real3x3> send_buf,ArrayView<Real3x3> recv_buf,Int32 rank) =0;
  virtual void gather(ConstArrayView<HPReal> send_buf,ArrayView<HPReal> recv_buf,Int32 rank) =0;
  //@}

  //! @name allGather variable
  //@{

  /*!
   * \brief Effectue un regroupement sur tous les processeurs.
   *
   * Il s'agit d'une opération collective. Le nombre d'éléments du tableau
   * \a send_buf peut être différent pour chaque processeur. Le tableau
   * \a recv_buf contient en sortie la concaténation des tableaux \a send_buf
   * de chaque processeur. Ce tableau \a recv_buf est éventuellement redimensionné
   * pour le processeurs de rang \a rank.
   */
  virtual void gatherVariable(ConstArrayView<char> send_buf,
                              Array<char>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<signed char> send_buf,
                              Array<signed char>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<unsigned char> send_buf,
                              Array<unsigned char>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<short> send_buf,
                              Array<short>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<unsigned short> send_buf,
                              Array<unsigned short>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<int> send_buf,
                              Array<int>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<unsigned int> send_buf,
                              Array<unsigned int>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<long> send_buf,
                              Array<long>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<unsigned long> send_buf,
                              Array<unsigned long>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<long long> send_buf,
                              Array<long long>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<unsigned long long> send_buf,
                              Array<unsigned long long>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<float> send_buf,
                              Array<float>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<double> send_buf,
                              Array<double>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<long double> send_buf,
                              Array<long double>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<APReal> send_buf,
                              Array<APReal>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<Real2> send_buf,
                              Array<Real2>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<Real3> send_buf,
                              Array<Real3>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<Real2x2> send_buf,
                              Array<Real2x2>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<Real3x3> send_buf,
                              Array<Real3x3>& recv_buf,Int32 rank) =0;
  virtual void gatherVariable(ConstArrayView<HPReal> send_buf,
                              Array<HPReal>& recv_buf,Int32 rank) =0;
  //@}

  //! @name allGather variable
  //@{

  /*!
   * \brief Effectue un regroupement sur tous les processeurs.
   *
   * Il s'agit d'une opération collective. Le nombre d'éléments du tableau
   * \a send_buf peut être différent pour chaque processeur. Le tableau
   * \a recv_buf contient en sortie la concaténation des tableaux \a send_buf
   * de chaque processeur. Ce tableau \a recv_buf est éventuellement redimensionné.
   */
  virtual void allGatherVariable(ConstArrayView<char> send_buf,
                                 Array<char>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<signed char> send_buf,
                                 Array<signed char>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<unsigned char> send_buf,
                                 Array<unsigned char>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<short> send_buf,
                                 Array<short>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<unsigned short> send_buf,
                                 Array<unsigned short>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<int> send_buf,
                                 Array<int>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<unsigned int> send_buf,
                                 Array<unsigned int>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<long> send_buf,
                                 Array<long>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<unsigned long> send_buf,
                                 Array<unsigned long>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<long long> send_buf,
                                 Array<long long>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<unsigned long long> send_buf,
                                 Array<unsigned long long>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<float> send_buf,
                                 Array<float>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<double> send_buf,
                                 Array<double>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<long double> send_buf,
                                 Array<long double>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<APReal> send_buf,
                                 Array<APReal>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<Real2> send_buf,
                                 Array<Real2>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<Real3> send_buf,
                                 Array<Real3>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<Real2x2> send_buf,
                                 Array<Real2x2>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<Real3x3> send_buf,
                                 Array<Real3x3>& recv_buf) =0;
  virtual void allGatherVariable(ConstArrayView<HPReal> send_buf,
                                 Array<HPReal>& recv_buf) =0;
  //@}

  //! @name opérations de réduction sur un scalaire
  //@{
  /*!
   * \brief Scinde un tableau sur plusieurs processeurs.
   */
  virtual void scatterVariable(ConstArrayView<char> send_buf,
                               ArrayView<char> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<signed char> send_buf,
                               ArrayView<signed char> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<unsigned char> send_buf,
                               ArrayView<unsigned char> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<short> send_buf,
                               ArrayView<short> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<unsigned short> send_buf,
                               ArrayView<unsigned short> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<int> send_buf,
                               ArrayView<int> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<unsigned int> send_buf,
                               ArrayView<unsigned int> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<long> send_buf,
                               ArrayView<long> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<unsigned long> send_buf,
                               ArrayView<unsigned long> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<long long> send_buf,
                               ArrayView<long long> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<unsigned long long> send_buf,
                               ArrayView<unsigned long long> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<float> send_buf,
                               ArrayView<float> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<double> send_buf,
                               ArrayView<double> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<long double> send_buf,
                               ArrayView<long double> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<APReal> send_buf,
                               ArrayView<APReal> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<Real2> send_buf,
                               ArrayView<Real2> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<Real3> send_buf,
                               ArrayView<Real3> recv_buf,Integer root) =0; 
  virtual void scatterVariable(ConstArrayView<Real2x2> send_buf,
                               ArrayView<Real2x2> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<Real3x3> send_buf,
                               ArrayView<Real3x3> recv_buf,Integer root) =0;
  virtual void scatterVariable(ConstArrayView<HPReal> send_buf,
                               ArrayView<HPReal> recv_buf,Integer root) =0;
  //@}

  //! @name opérations de réduction sur un scalaire
  //@{
  /*!
   * \brief Effectue la réduction de type \a rt sur le réel \a v et retourne la valeur
   */
  virtual char reduce(eReduceType rt,char v) =0;
  virtual signed char reduce(eReduceType rt,signed char v) =0;
  virtual unsigned char reduce(eReduceType rt,unsigned char v) =0;
  virtual short reduce(eReduceType rt,short v) =0;
  virtual unsigned short reduce(eReduceType rt,unsigned short v) =0;
  virtual int reduce(eReduceType rt,int v) =0;
  virtual unsigned int reduce(eReduceType rt,unsigned int v) =0;
  virtual long reduce(eReduceType rt,long v) =0;
  virtual unsigned long reduce(eReduceType rt,unsigned long v) =0;
  virtual long long reduce(eReduceType rt,long long v) =0;
  virtual unsigned long long reduce(eReduceType rt,unsigned long long v) =0;
  virtual float reduce(eReduceType rt,float v) =0;
  virtual double reduce(eReduceType rt,double v) =0;
  virtual long double reduce(eReduceType rt,long double v) =0;
  virtual APReal reduce(eReduceType rt,APReal v) =0;
  virtual Real2 reduce(eReduceType rt,Real2 v) =0;
  virtual Real3 reduce(eReduceType rt,Real3 v) =0;
  virtual Real2x2 reduce(eReduceType rt,Real2x2 v) =0;
  virtual Real3x3 reduce(eReduceType rt,Real3x3 v) =0;
  virtual HPReal reduce(eReduceType rt,HPReal v) =0;
  //@}

  //! @name opérations de réduction sur un scalaire
  //@{
  /*!
   * \brief Calcule en une opération la somme, le min, le max d'une valeur.
   *
   * Calcule le minimum, le maximum et la somme de la valeur \a val.
   * \param val valeur servant pour le calcul
   * \param[out] min_val valeur minimale
   * \param[out] max_val valeur maximale
   * \param[out] sum_val somme des valeurs
   * \param[out] min_rank rang du processeur ayant la valeur minimale
   * \param[out] max_rank rang du processeur ayant la valeur maximale
   */
  virtual void computeMinMaxSum(char val,char& min_val,
                                char& max_val,char& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(signed char val,signed char& min_val,
                                signed char& max_val,signed char& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(unsigned char val,unsigned char& min_val,
                                unsigned char& max_val,unsigned char& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(short val,short& min_val,
                                short& max_val,short& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(unsigned short val,unsigned short& min_val,
                                unsigned short& max_val,unsigned short& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(int val,int& min_val,
                                int& max_val,int& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(unsigned int val,unsigned int& min_val,
                                unsigned int& max_val,unsigned int& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(long val,long& min_val,
                                long& max_val,long& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(unsigned long val,unsigned long& min_val,
                                unsigned long& max_val,unsigned long& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(long long val,long long& min_val,
                                long long& max_val,long long& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(unsigned long long val,unsigned long long& min_val,
                                unsigned long long& max_val,unsigned long long& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(float val,float& min_val,
                                float& max_val,float& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(double val,double& min_val,
                                double& max_val,double& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(long double val,long double& min_val,
                                long double& max_val,long double& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(APReal val,APReal& min_val,
                                APReal& max_val,APReal& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(Real2 val,Real2& min_val,
                                Real2& max_val,Real2& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(Real3 val,Real3& min_val,
                                Real3& max_val,Real3& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(Real2x2 val,Real2x2& min_val,
                                Real2x2& max_val,Real2x2& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(Real3x3 val,Real3x3& min_val,
                                Real3x3& max_val,Real3x3& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  virtual void computeMinMaxSum(HPReal val,HPReal& min_val,
                                HPReal& max_val,HPReal& sum_val,
                                Int32& min_rank,Int32& max_rank) =0;
  //@}

  //! @name opérations de réduction sur un vecteur
  //@{
  /*!
   * \brief Calcule en une opération la somme, le min, le max d'une valeur.
   *
   * Calcule le minimum, le maximum et la somme de la valeur \a val.
   * \param val valeur servant pour le calcul
   * \param[out] min_val valeur minimale
   * \param[out] max_val valeur maximale
   * \param[out] sum_val somme des valeurs
   * \param[out] min_rank rang du processeur ayant la valeur minimale
   * \param[out] max_rank rang du processeur ayant la valeur maximale
   */
  virtual void computeMinMaxSum(ConstArrayView<char> values, ArrayView<char> min_values,
                                ArrayView<char> max_values, ArrayView<char> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<signed char> values, ArrayView<signed char> min_values,
                                ArrayView<signed char> max_values, ArrayView<signed char> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned char> values, ArrayView<unsigned char> min_values,
                                ArrayView<unsigned char> max_values, ArrayView<unsigned char> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<short> values, ArrayView<short> min_values,
                                ArrayView<short> max_values, ArrayView<short> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned short> values, ArrayView<unsigned short> min_values,
                                ArrayView<unsigned short> max_values, ArrayView<unsigned short> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<int> values, ArrayView<int> min_values,
                                ArrayView<int> max_values, ArrayView<int> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned int> values, ArrayView<unsigned int> min_values,
                                ArrayView<unsigned int> max_values, ArrayView<unsigned int> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<long> values, ArrayView<long> min_values,
                                ArrayView<long> max_values, ArrayView<long> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned long> values, ArrayView<unsigned long> min_values,
                                ArrayView<unsigned long> max_values, ArrayView<unsigned long> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<long long> values, ArrayView<long long> min_values,
                                ArrayView<long long> max_values, ArrayView<long long> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<unsigned long long> values, ArrayView<unsigned long long> min_values,
                                ArrayView<unsigned long long> max_values, ArrayView<unsigned long long> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<float> values, ArrayView<float> min_values,
                                ArrayView<float> max_values, ArrayView<float> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<double> values, ArrayView<double> min_values,
                                ArrayView<double> max_values, ArrayView<double> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<long double> values, ArrayView<long double> min_values,
                                ArrayView<long double> max_values, ArrayView<long double> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<APReal> values, ArrayView<APReal> min_values,
                                ArrayView<APReal> max_values, ArrayView<APReal> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<Real2> values, ArrayView<Real2> min_values,
                                ArrayView<Real2> max_values, ArrayView<Real2> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<Real3> values, ArrayView<Real3> min_values,
                                ArrayView<Real3> max_values, ArrayView<Real3> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<Real2x2> values, ArrayView<Real2x2> min_values,
                                ArrayView<Real2x2> max_values, ArrayView<Real2x2> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<Real3x3> values, ArrayView<Real3x3> min_values,
                                ArrayView<Real3x3> max_values, ArrayView<Real3x3> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  virtual void computeMinMaxSum(ConstArrayView<HPReal> values, ArrayView<HPReal> min_values,
                                ArrayView<HPReal> max_values, ArrayView<HPReal> sum_values,
                                ArrayView<Int32> min_ranks, ArrayView<Int32> max_ranks) =0;
  //@}


  //! @name opérations de réduction sur un tableau
  //@{
  /*!
   * \brief Effectue la réduction de type \a rt sur le tableau \a v.
   */
  virtual void reduce(eReduceType rt,ArrayView<char> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<signed char> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<unsigned char> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<short> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<unsigned short> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<int> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<unsigned int> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<long> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<unsigned long> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<long long> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<unsigned long long> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<float> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<double> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<long double> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<APReal> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<Real2> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<Real3> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<Real2x2> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<Real3x3> v) =0;
  virtual void reduce(eReduceType rt,ArrayView<HPReal> v) =0;
  //@}

  /*!
   * @name opérations de broadcast
   *
   * \brief Envoie un tableau de valeurs sur tous les sous-domaines.
   *
   * Cette opération envoie le tableau de valeur \a send_buf sur tous
   * les sous-domaines. Le tableau utilisé est celui dont le rang (commRank) est \a rank.
   * Tous les sous-domaines participants doivent appelés cette méthode avec
   * le même paramètre \a rank et avoir un tableau \a send_buf
   * contenant le même nombre d'éléments.
   */
  //@{
  virtual void broadcast(ArrayView<char> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<signed char> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<unsigned char> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<short> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<unsigned short> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<int> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<unsigned int> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<long> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<unsigned long> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<long long> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<unsigned long long> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<float> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<double> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<long double> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<APReal> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<Real2> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<Real3> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<Real2x2> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<Real3x3> send_buf,Int32 rank) =0;
  virtual void broadcast(ArrayView<HPReal> send_buf,Int32 rank) =0;
  virtual void broadcastString(String& str,Int32 rank) =0;

  virtual void broadcastSerializer(ISerializer* values,Int32 rank) =0;
  /*! \brief Effectue un broadcast d'une zone mémoire.
   *
   * Le processeur qui effectue le broadcast est donnée par \id. Le tableau
   * envoyé est alors donnée par \a bytes. Les processeurs réceptionnent
   * le tableau dans \a bytes. Ce tableau est alloué automatiquement, les processeurs
   * réceptionnant n'ont pas besoin de connaitre le nombre d'octets devant être envoyés.
   *
   */
  virtual void broadcastMemoryBuffer(ByteArray& bytes,Int32 rank) =0;
  //@}

  /*!
   * @name opérations d'envoie de messages
   *
   * \brief Envoie bloquant d'un tableau de valeurs à un sous-domaine.
   *
   * Envoie les valeurs du tableau \a values au sous-domaine \a rank.
   * Le sous-domaine doit effectuer une réception correspondante (le numéro
   * de sous-domaine doit être celui de ce gestionnaire et le type et la
   * taille du tableau doit correspondre) avec la fonction recvValues().
   * L'envoie est bloquant.
   */
  //@{
  virtual void send(ConstArrayView<char> values,Int32 rank) =0;
  virtual void send(ConstArrayView<signed char> values,Int32 rank) =0;
  virtual void send(ConstArrayView<unsigned char> values,Int32 rank) =0;
  virtual void send(ConstArrayView<short> values,Int32 rank) =0;
  virtual void send(ConstArrayView<unsigned short> values,Int32 rank) =0;
  virtual void send(ConstArrayView<int> values,Int32 rank) =0;
  virtual void send(ConstArrayView<unsigned int> values,Int32 rank) =0;
  virtual void send(ConstArrayView<long> values,Int32 rank) =0;
  virtual void send(ConstArrayView<unsigned long> values,Int32 rank) =0;
  virtual void send(ConstArrayView<long long> values,Int32 rank) =0;
  virtual void send(ConstArrayView<unsigned long long> values,Int32 rank) =0;
  virtual void send(ConstArrayView<float> values,Int32 rank) =0;
  virtual void send(ConstArrayView<double> values,Int32 rank) =0;
  virtual void send(ConstArrayView<long double> values,Int32 rank) =0;
  virtual void send(ConstArrayView<APReal> values,Int32 rank) =0;
  virtual void send(ConstArrayView<Real2> values,Int32 rank) =0;
  virtual void send(ConstArrayView<Real3> values,Int32 rank) =0;
  virtual void send(ConstArrayView<Real2x2> values,Int32 rank) =0;
  virtual void send(ConstArrayView<Real3x3> values,Int32 rank) =0;
  virtual void send(ConstArrayView<HPReal> values,Int32 rank) =0;

  virtual void sendSerializer(ISerializer* values,Int32 rank) =0;
  /*!
   * la requête donnée en retour doit être utilisée dans waitAllRequests() ou
   * libérée par appel à freeRequests().
   */
  ARCCORE_DEPRECATED_2019("Use createSendSerializer(Int32 rank) instead")
  virtual Parallel::Request sendSerializer(ISerializer* values,Int32 rank,ByteArray& bytes) =0;

  /*!
   * \brief Créé un message non bloquant pour envoyer des données sérialisées au rang \a rank.
   *
   * Le message est traité uniquement lors de l'appel à processMessages().
   */
  virtual ISerializeMessage* createSendSerializer(Int32 rank) =0;
  //@}

  //! @name opérations de réception de messages.
  //@{
  /*! Reception du rang \a rank le tableau \a values */
  virtual void recv(ArrayView<char> values,Int32 rank) =0;
  virtual void recv(ArrayView<signed char> values,Int32 rank) =0;
  virtual void recv(ArrayView<unsigned char> values,Int32 rank) =0;
  virtual void recv(ArrayView<short> values,Int32 rank) =0;
  virtual void recv(ArrayView<unsigned short> values,Int32 rank) =0;
  virtual void recv(ArrayView<int> values,Int32 rank) =0;
  virtual void recv(ArrayView<unsigned int> values,Int32 rank) =0;
  virtual void recv(ArrayView<long> values,Int32 rank) =0;
  virtual void recv(ArrayView<unsigned long> values,Int32 rank) =0;
  virtual void recv(ArrayView<long long> values,Int32 rank) =0;
  virtual void recv(ArrayView<unsigned long long> values,Int32 rank) =0;
  virtual void recv(ArrayView<float> values,Int32 rank) =0;
  virtual void recv(ArrayView<double> values,Int32 rank) =0;
  virtual void recv(ArrayView<long double> values,Int32 rank) =0;
  virtual void recv(ArrayView<APReal> values,Int32 rank) =0;
  virtual void recv(ArrayView<Real2> values,Int32 rank) =0;
  virtual void recv(ArrayView<Real3> values,Int32 rank) =0;
  virtual void recv(ArrayView<Real2x2> values,Int32 rank) =0;
  virtual void recv(ArrayView<Real3x3> values,Int32 rank) =0;
  virtual void recv(ArrayView<HPReal> values,Int32 rank) =0;
  virtual void recvSerializer(ISerializer* values,Int32 rank) =0;
  //@}

  /*!
   * \brief Créé un message non bloquant pour recevoir des données sérialisées du rang \a rank.
   *
   * Le message est traité uniquement lors de l'appel à processMessages().
   */
  virtual ISerializeMessage* createReceiveSerializer(Int32 rank) =0;

  /*!
   * \brief Exécute les opérations des messages \a messages
   */
  virtual void processMessages(ConstArrayView<ISerializeMessage*> messages) =0;

  /*!
   * \brief Exécute les opérations des messages \a messages
   */
  virtual void processMessages(ConstArrayView<Ref<ISerializeMessage>> messages) =0;

  /*!
   * \brief Libère les requêtes.
   */
  virtual void freeRequests(ArrayView<Parallel::Request> requests) =0;

  /*! @name opérations d'envoie de messages non bloquants
   *
   * \brief Envoie un tableau de valeurs à un rang \a rank.
   *
   * Envoie les valeurs du tableau \a values à l'instance de rang \a rank.
   * Le destinataire doit effectuer une réception correspondante (dont le
   * rang doit être celui de ce gestionnaire et le type et la
   * taille du tableau doit correspondre) avec la fonction recvValues().
   * L'envoie est bloquant si \a is_blocking vaut \a true, non bloquant s'il vaut \a false.
   * Dans ce dernier cas, la requête retournée doit être utilisée dans waitAllRequests()
   * ou libérée par freeRequests().
   */
  //@{
  virtual Request send(ConstArrayView<char> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<signed char> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<unsigned char> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<short> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<unsigned short> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<int> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<unsigned int> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<long> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<unsigned long> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<long long> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<unsigned long long> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<float> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<double> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<long double> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<APReal> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<Real2> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<Real3> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<Real2x2> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<Real3x3> values,Int32 rank,bool is_blocking) =0;
  virtual Request send(ConstArrayView<HPReal> values,Int32 rank,bool is_blocking) =0;
  //@}

  //! @name opérations de réception de messages non bloquantes.
  //@{
  /*! Recoie du sous-domaine \a rank le tableau \a values */
  virtual Request recv(ArrayView<char> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<signed char> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<unsigned char> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<short> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<unsigned short> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<int> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<unsigned int> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<long> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<unsigned long> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<long long > values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<unsigned long long> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<float> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<double> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<long double> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<APReal> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<Real2> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<Real3> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<Real2x2> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<Real3x3> values,Int32 rank,bool is_blocking) =0;
  virtual Request recv(ArrayView<HPReal> values,Int32 rank,bool is_blocking) =0;
  //@}

  //! @name opérations de réception génériques de messages 
  //@{
  /*! Reception du message \a message le tableau \a values */
  virtual Request receive(Span<char> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<signed char> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<unsigned char> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<short> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<unsigned short> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<int> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<unsigned int> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<long> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<unsigned long> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<long long> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<unsigned long long> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<float> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<double> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<long double> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<APReal> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<Real2> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<Real3> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<Real2x2> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<Real3x3> values,const PointToPointMessageInfo& message) =0;
  virtual Request receive(Span<HPReal> values,const PointToPointMessageInfo& message) =0;
  virtual Request receiveSerializer(ISerializer* values,const PointToPointMessageInfo& message) =0;
  //@}

  //! @name opérations d'envoie génériques de messages 
  //@{
  /*! Envoie du message \a message avec les valeurs du tableau \a values */
  virtual Request send(Span<const char> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const signed char> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const unsigned char> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const short> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const unsigned short> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const int> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const unsigned int> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const long> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const unsigned long> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const long long> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const unsigned long long> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const float> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const double> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const long double> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const APReal> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const Real2> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const Real3> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const Real2x2> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const Real3x3> values,const PointToPointMessageInfo& message) =0;
  virtual Request send(Span<const HPReal> values,const PointToPointMessageInfo& message) =0;
  virtual Request sendSerializer(const ISerializer* values,const PointToPointMessageInfo& message) =0;
  //@}

  /*!
   * \brief Sonde si des messages sont disponibles.
   *
   * \sa Arccore::MessagePassing::mpProbe().
   */
  virtual MessageId probe(const PointToPointMessageInfo& message) =0;

  /*!
   * \brief Sonde si des messages sont disponibles.
   *
   * \sa Arccore::MessagePassing::mpLegacyProbe().
   */
  virtual MessageSourceInfo legacyProbe(const PointToPointMessageInfo& message) =0;

  virtual void sendRecv(ConstArrayView<char> send_buf,
                        ArrayView<char> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<signed char> send_buf,
                        ArrayView<signed char> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<unsigned char> send_buf,
                        ArrayView<unsigned char> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<short> send_buf,
                        ArrayView<short> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<unsigned short> send_buf,
                        ArrayView<unsigned short> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<int> send_buf,
                        ArrayView<int> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<unsigned int> send_buf,
                        ArrayView<unsigned int> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<long> send_buf,
                        ArrayView<long> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<unsigned long> send_buf,
                        ArrayView<unsigned long> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<long long> send_buf,
                        ArrayView<long long> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<unsigned long long> send_buf,
                        ArrayView<unsigned long long> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<float> send_buf,
                        ArrayView<float> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<double> send_buf,
                        ArrayView<double> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<long double> send_buf,
                        ArrayView<long double> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<APReal> send_buf,
                        ArrayView<APReal> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<Real2> send_buf,
                        ArrayView<Real2> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<Real3> send_buf,
                        ArrayView<Real3> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<Real2x2> send_buf,
                        ArrayView<Real2x2> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<Real3x3> send_buf,
                        ArrayView<Real3x3> recv_buf,Int32 rank) =0;
  virtual void sendRecv(ConstArrayView<HPReal> send_buf,
                        ArrayView<HPReal> recv_buf,Int32 rank) =0;

  virtual void allToAll(ConstArrayView<char> send_buf,ArrayView<char> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<signed char> send_buf,ArrayView<signed char> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<unsigned char> send_buf,ArrayView<unsigned char> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<short> send_buf,ArrayView<short> recv_buf,Integer count) =0;
  virtual void allToAll(ConstArrayView<unsigned short> send_buf,ArrayView<unsigned short> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<int> send_buf,ArrayView<int> recv_buf,Integer count) =0;
  virtual void allToAll(ConstArrayView<unsigned int> send_buf,ArrayView<unsigned int> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<long> send_buf,ArrayView<long> recv_buf,Integer count) =0;
  virtual void allToAll(ConstArrayView<unsigned long> send_buf,ArrayView<unsigned long> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<long long> send_buf,ArrayView<long long> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<unsigned long long> send_buf,
                        ArrayView<unsigned long long> recv_buf,Integer count) =0;
  virtual void allToAll(ConstArrayView<float> send_buf,ArrayView<float> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<double> send_buf,ArrayView<double> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<long double> send_buf,ArrayView<long double> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<APReal> send_buf,ArrayView<APReal> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<Real2> send_buf,ArrayView<Real2> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<Real3> send_buf,ArrayView<Real3> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<Real2x2> send_buf,ArrayView<Real2x2> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<Real3x3> send_buf,ArrayView<Real3x3> recv_buf,
                        Integer count) =0;
  virtual void allToAll(ConstArrayView<HPReal> send_buf,ArrayView<HPReal> recv_buf,
                        Integer count) =0;


  /*! @name allToAll variable
   *
   * \brief Effectue un allToAll variable
   */
  //@{
  virtual void allToAllVariable(ConstArrayView<char> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<char> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<signed char> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<signed char> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<unsigned char> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<unsigned char> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<short> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<short> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<unsigned short> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<unsigned short> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<int> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<int> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<unsigned int> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<unsigned int> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<long> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<long> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<unsigned long> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<unsigned long> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<long long> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<long long> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<unsigned long long> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<unsigned long long> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<float> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<float> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<double> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<double> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<long double> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<long double> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<APReal> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<APReal> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<Real2> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<Real2> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<Real3> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<Real3> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<Real2x2> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<Real2x2> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<Real3x3> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<Real3x3> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  virtual void allToAllVariable(ConstArrayView<HPReal> send_buf,Int32ConstArrayView send_count,
                                Int32ConstArrayView send_index,ArrayView<HPReal> recv_buf,
                                Int32ConstArrayView recv_count,Int32ConstArrayView recv_index) =0;
  //@}

  /*! @name scan
   *
   * \brief Effectue un algorithme de scan équivalent en sémantique à MPI_Scan
   */
  //@{
  //! Applique un algorithme de prefix-um sur les valeurs de \a v via l'opération \a rt.
  virtual void scan(eReduceType rt,ArrayView<char> v) =0;
  virtual void scan(eReduceType rt,ArrayView<signed char> v) =0;
  virtual void scan(eReduceType rt,ArrayView<unsigned char> v) =0;
  virtual void scan(eReduceType rt,ArrayView<short> v) =0;
  virtual void scan(eReduceType rt,ArrayView<unsigned short> v) =0;
  virtual void scan(eReduceType rt,ArrayView<int> v) =0;
  virtual void scan(eReduceType rt,ArrayView<unsigned int> v) =0;
  virtual void scan(eReduceType rt,ArrayView<long> v) =0;
  virtual void scan(eReduceType rt,ArrayView<unsigned long> v) =0;
  virtual void scan(eReduceType rt,ArrayView<long long> v) =0;
  virtual void scan(eReduceType rt,ArrayView<unsigned long long> v) =0;
  virtual void scan(eReduceType rt,ArrayView<float> v) =0;
  virtual void scan(eReduceType rt,ArrayView<double> v) =0;
  virtual void scan(eReduceType rt,ArrayView<long double> v) =0;
  virtual void scan(eReduceType rt,ArrayView<APReal> v) =0;
  virtual void scan(eReduceType rt,ArrayView<Real2> v) =0;
  virtual void scan(eReduceType rt,ArrayView<Real3> v) =0;
  virtual void scan(eReduceType rt,ArrayView<Real2x2> v) =0;
  virtual void scan(eReduceType rt,ArrayView<Real3x3> v) =0;
  virtual void scan(eReduceType rt,ArrayView<HPReal> v) =0;
  //@}

  /*!
   * \brief Créé une liste pour gérer les 'ISerializeMessage'.
   *
   * \deprecated Utiliser createSerializeMessageListRef() à la place.
   */
  ARCCORE_DEPRECATED_2020("Use createSerializeMessageListRef() instead")
  virtual ISerializeMessageList* createSerializeMessageList() =0;

  //! Créé une liste pour gérer les 'ISerializeMessage'
  virtual Ref<ISerializeMessageList> createSerializeMessageListRef() =0;

  //! @name opérations de synchronisation et opérations asynchrones
  //@{
  //! Effectue une barière
  virtual void barrier() =0;

  //! Bloque en attendant que les requêtes \a rvalues soient terminées
  virtual void waitAllRequests(ArrayView<Request> rvalues) =0;

  /*!
  * \brief Bloque en attendant qu'une des requêtes \a rvalues soit terminée.
  *
  * Retourne un tableau d'indices des requêtes réalisées.
  */
  virtual UniqueArray<Integer> waitSomeRequests(ArrayView<Request> rvalues) =0;

  /*!
  * \brief Test si une des requêtes \a rvalues est terminée.
  *
  * Retourne un tableau d'indices des requêtes réalisées.
  */
  virtual UniqueArray<Integer> testSomeRequests(ArrayView<Request> rvalues) =0;

 //@}


  //! @name opérations diverses
  //@{

  /*!
   * \brief Retourne un gestionnaire de parallélisme séquentiel.
   *
   * Cette instance reste propriétaire de l'instance retournée qui ne doit
   * pas être détruite. La durée de vie de l'instance retournée est
   * la même que cette instance.
   */
  virtual IParallelMng* sequentialParallelMng() =0;
  virtual Ref<IParallelMng> sequentialParallelMngRef() =0;

  //@}

  /*!
   * \brief Retourne une opération pour récupérer les valeurs d'une variable
   * sur les entités d'un autre sous-domaine.
   *
   * L'instance retournée doit être détruite par l'opérateur delete.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createGetVariablesValuesOperationRef() instead")]]
  virtual IGetVariablesValuesParallelOperation* createGetVariablesValuesOperation() =0;

  /*!
   * \brief Retourne une opération pour transférer des valeurs
   * entre sous-domaine.
   *
   * L'instance retournée doit être détruite par l'opérateur delete.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createTransferValuesOperationRef() instead")]]
  virtual ITransferValuesParallelOperation* createTransferValuesOperation() =0;

  /*!
   * \brief Retourne une interface pour transférer des messages
   * entre processeurs.
   *
   * L'instance retournée doit être détruite par l'opérateur delete.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createExchangerRef() instead")]]
  virtual IParallelExchanger* createExchanger() =0;

  /*!
   * \brief Retourne une interface pour synchroniser des
   * variables sur le groupe de la famille \a family
   *
   * L'instance retournée doit être détruite par l'opérateur delete.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createSynchronizerRef() instead")]]
  virtual IVariableSynchronizer* createSynchronizer(IItemFamily* family) =0;
 
  /*!
   * \brief Retourne une interface pour synchroniser des
   * variables sur le groupe \a group.
   *
   * L'instance retournée doit être détruite par l'opérateur delete.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createSynchronizerRef() instead")]]
  virtual IVariableSynchronizer* createSynchronizer(const ItemGroup& group) =0;

  /*!
   * \brief Créé une instance contenant les infos sur la topologie des rangs de ce gestionnnaire.
   *
   * Cette opération est collective.
   *
   * L'instance retournée doit être détruite par l'opérateur delete.
   */
  [[deprecated("Y2021: Use Arcane::ParallelMngUtils;:createTopologyRef() instead")]]
  virtual IParallelTopology* createTopology() =0;

  /*!
   * \brief Informations sur la réplication.
   *
   * Le pointeur retourné n'est jamais nul et reste la propriété de cette
   * instance.
   */
  virtual IParallelReplication* replication() const =0;

  /*!
   * \internal
   * \brief Positionne les Informations sur la réplication.
   *
   * Cette méthode est interne à Arcane et ne doit être appelée que lors de l'initialisation.
   */
  virtual void setReplication(IParallelReplication* v) =0;

  /*!
   * \brief Créé un nouveau gestionnaire de parallélisme pour un sous-ensemble
   * des rangs.
   *
   * \deprecated Utiliser createSubParallelMngRef() à la place
   */
  ARCCORE_DEPRECATED_2020("Use createSubParallelMngRef() instead")
  virtual IParallelMng* createSubParallelMng(Int32ConstArrayView kept_ranks) =0;

  /*!
   * \brief Créé un nouveau gestionnaire de parallélisme pour un sous-ensemble
   * des rangs.
   *
   * Cette opération est collective.
   *
   * Cett opération permet de créér un nouveau gestionnaire contenant
   * uniquement les rangs \a kept_ranks de ce gestionnaire.
   *
   * Si le rang appelant cette opération n'est pas dans \a kept_ranks,
   * retourne 0.
   *
   * L'instance retournée doit être détruire par l'opérateur delete.
   */
  virtual Ref<IParallelMng> createSubParallelMngRef(Int32ConstArrayView kept_ranks) =0;

  /*!
   * \brief Créé une liste de requêtes pour ce gestionnaire.
   */
  virtual Ref<Parallel::IRequestList> createRequestListRef() =0;

  //! Gestionnaire des statistiques
  virtual IStat* stat() =0;

  //! Affiche des statistiques liées à ce gestionnaire du parallélisme
  virtual void printStats() =0;

  //! Interface des opérations collectives non blocantes.
  virtual IParallelNonBlockingCollective* nonBlockingCollective() const =0;

  //! Gestionnaire de message de %Arccore associé
  virtual IMessagePassingMng* messagePassingMng() const =0;

 public:

  //! API interne à Arcane
  virtual IParallelMngInternal* _internalApi() =0;

 private:

  /*!
   * \internal
   * \brief Fabrique des fonctions utilitaires.
   */
  virtual Ref<IParallelMngUtilsFactory> _internalUtilsFactory() const =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'un conteneur de 'IParallelMng'.
 *
 * Un conteneur de IParallelMng gère en mode mémoire partagée un ensemble
 * de IParallelMng d'un même communicateur.
 *
 * \note Ne pas utiliser en dehors d'Arcane. API non stabilisée.
 */
class ARCANE_CORE_EXPORT IParallelMngContainer
{
  ARCCORE_DECLARE_REFERENCE_COUNTED_INCLASS_METHODS();
 protected:
  virtual ~IParallelMngContainer() = default;
 public:
  //! Créé le IParallelMng pour le rang local \a local_rank
  virtual Ref<IParallelMng> _createParallelMng(Int32 local_rank,ITraceMng* tm) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Interface d'une fabrique de conteneur de 'IParallelMng'.
 * \note Ne pas utiliser en dehors d'Arcane. API non stabilisée.
 */
class ARCANE_CORE_EXPORT IParallelMngContainerFactory
{
 public:
  virtual ~IParallelMngContainerFactory() = default;
 public:
  /*!
   * \brief Créé un conteneur pour \a nb_local_rank rangs locaux et
   * avec comme communicateur \a communicator.
   *
   * Le communicateur MPI \a communicator peut être nul en mode séquentiel ou
   * mémoire partagé.
   * Le nombre de rangs locaux vaut 1 en mode séquentiel ou en mode MPI pure.
   *
   * Le second communicateur \a machine_communicator est utile qu'en mode
   * hydride.
   * Dans les autres modes, il peut être nul.
   */
  virtual Ref<IParallelMngContainer>
  _createParallelMngBuilder(Int32 nb_local_rank, Parallel::Communicator communicator,
                            Parallel::Communicator machine_communicator) =0;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namepsace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif  
