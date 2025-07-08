// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcanePostProcessingModule.cc                               (C) 2000-2025 */
/*                                                                           */
/* Module de post-traitement.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/Ptr.h"
#include "arcane/utils/List.h"

#include "arcane/core/EntryPoint.h"
#include "arcane/core/ISubDomain.h"
#include "arcane/core/IVariableMng.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/ItemGroup.h"
#include "arcane/core/Directory.h"
#include "arcane/core/ITimeHistoryMng.h"
#include "arcane/core/ServiceUtils.h"
#include "arcane/core/IPostProcessorWriter.h"
#include "arcane/core/MeshAccessor.h"
#include "arcane/core/IMesh.h"
#include "arcane/core/VariableTypes.h"
#include "arcane/core/CommonVariables.h"
#include "arcane/core/MathUtils.h"
#include "arcane/core/ITimeLoopMng.h"
#include "arcane/core/ItemEnumerator.h"
#include "arcane/core/ModuleFactory.h"
#include "arcane/core/Timer.h"
#include "arcane/core/VariableCollection.h"
#include "arcane/core/OutputChecker.h"

#include "arcane/std/ArcanePostProcessing_axl.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de sortie pour le dépouillement.
 *
 * Lorsque ce module est connecté, ce module gère les sorties pour
 * le dépouillement.
 *
 * Si aucune variable n'est spécifiée, aucune sortie n'est effectuée. Le
 * champ #m_do_output est alors à faux.
 */
class ArcanePostProcessingModule
: public ArcaneArcanePostProcessingObject
{
 public:

  explicit ArcanePostProcessingModule(const ModuleBuildInfo& mbi);
  ~ArcanePostProcessingModule() override;

 public:

  VersionInfo versionInfo() const override { return VersionInfo(0, 1, 2); }

 public:

  void exportData() override;
  void exportDataStart() override;

  void postProcessingStartInit() override;
  void postProcessingInit() override;
  void postProcessingExit() override;

 private:

  OutputChecker m_output_checker;
  OutputChecker m_history_output_checker;
  VariableArrayReal m_times; //!< Instants de temps des sauvegardes
  bool m_is_output_active = true; //!< \a true si les sorties sont actives
  //! Indique si on réalise des sorties lors de cette itération
  bool m_is_output_at_current_iteration = false;
  Directory m_output_directory; //!< Répertoire de sortie
  bool m_output_dir_created = false; //!< \a true si répertoire créé.
  VariableList m_variables;    //!< Liste des variables a exporter
  ItemGroupList m_groups; //!< Liste des groupes à exporter
  Timer* m_post_processor_timer = nullptr; //!< Timer pour le temps passé à écrire

 private:

  void _readConfig();
  void _saveAtTime(Real);

  void _checkCreateOutputDir();
  void _markCurrentIterationPostProcessing();
  void _resetCurrentIterationPostProcessing();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ARCANEPOSTPROCESSING(ArcanePostProcessingModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcanePostProcessingModule::
ArcanePostProcessingModule(const ModuleBuildInfo& mbi)
: ArcaneArcanePostProcessingObject(mbi)
, m_output_checker(mbi.subDomain(),"PostProcessing")
, m_history_output_checker(mbi.subDomain(),"PostProcessingHistory")
, m_times(VariableBuilder(this,"ExportTimes"))
{
  m_output_checker.assignIteration(&m_next_iteration,&options()->outputPeriod);
  m_output_checker.assignGlobalTime(&m_next_global_time,&options()->outputFrequency);

  m_history_output_checker.assignIteration(&m_history_next_iteration,&options()->outputHistoryPeriod);
  m_post_processor_timer = new Timer(mbi.subDomain(),"PostProcessorTimer",Timer::TimerReal);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcanePostProcessingModule::
~ArcanePostProcessingModule()
{
  delete m_post_processor_timer;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcanePostProcessingModule::
_readConfig()
{
  Integer nb_var   = options()->output().variable.size();
  Integer nb_group = options()->output().group.size();

  IVariableMng* var_mng = subDomain()->variableMng();
  IMesh* mesh = subDomain()->defaultMesh();

  if (nb_var!=0){
    std::set<String> used_variables; // Liste des variables déjà indiquées
    m_variables.clear();
    info() << " ";
    info() << "-- List of output variables (" << nb_var << " variables):";
    for( Integer i=0; i<nb_var; ++i ){
      String varname(options()->output().variable[i]);
      IVariable* var = var_mng->findMeshVariable(defaultMesh(),varname);
      if (!var)
        ARCANE_FATAL("PostTreatment: no variable with name '{0}' exists",varname);
      eItemKind ik = var->itemKind();
      if (ik!=IK_Node && ik!=IK_Edge && ik!=IK_Face && ik!=IK_Cell)
        ARCANE_FATAL("PostTreatment: variable ({0}) must"
                     " be a mesh variable (node, edge, face or cell)",varname);

      if (used_variables.find(varname)==used_variables.end()){
        info() << "Variable <" << varname << ">";
        m_variables.add(var);
        used_variables.insert(varname);
        var->addTag(IVariable::TAG_POST_PROCESSING,"1");
      }
      else{
        warning() << "Variable <" << varname << "> required twice during post-processing analysis";
      }
    }
  }
  else
    m_is_output_active = false;

  if (nb_group!=0){
    std::set<String> used_groups; // Liste des groupes déjà indiquées
    //m_group_list.resize(nb_group);
    info() << " ";
    info() << "-- List of output groups (" << nb_group << " groups):";
    for( Integer i=0; i<nb_group; ++i ){
      ItemGroup group = options()->output().group[i];
      if (group.null())
        continue;
      String groupname = group.name();
      if (used_groups.find(groupname)==used_groups.end()){
        info() << "Group <" << groupname << ">";
        used_groups.insert(groupname);
        m_groups.add(group);
      }
      else
        warning() << "Group <" << groupname << "> required twice during post-processing analysis";
    }
    //m_group_list.resize(index);
  }
  else{
    // Si aucun groupe spécifié, sauve uniquement l'ensemble des mailles.
    //m_groups.resize(1);

    //! AMR
    if (mesh->isAmrActivated())
      m_groups.add(mesh->allActiveCells());
    else
      m_groups.add(mesh->allCells());
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcanePostProcessingModule::
_checkCreateOutputDir()
{
  if (m_output_dir_created)
    return;
  m_output_directory = Directory(subDomain()->exportDirectory(),"depouillement");
  m_output_directory.createDirectory();
  m_output_dir_created = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcanePostProcessingModule::
postProcessingInit()
{
  info() << " -------------------------------------------";
  info() << "|            POST PROCESSING                |";
  info() << " -------------------------------------------";
  info() << " ";

  bool is_continue = subDomain()->isContinue();

  info() << "Variables output:";
  m_output_checker.initialize(is_continue);

  info() << "History output:";
  m_history_output_checker.initialize(is_continue);

  _readConfig();

  // Positionnement de l'option 'shrink' du timeHistoryMng depuis l'axl
  if (options()->outputHistoryShrink)
    subDomain()->timeHistoryMng()->setShrinkActive(options()->outputHistoryShrink);

  // initialize parameter with a dry call to checker
  const CommonVariables& vc = subDomain()->commonVariables();
  const Real old_time = vc.globalOldTime();
  const Real current_time = vc.globalTime();
  m_output_checker.check(old_time, current_time, vc.globalIteration(), 0);
  m_history_output_checker.check(old_time, current_time, vc.globalIteration(), 0);

  if (options()->saveInit())
    _saveAtTime(current_time);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcanePostProcessingModule::
postProcessingStartInit()
{
  m_next_global_time = 0.0;
  m_next_iteration = 0;
  m_curves_next_global_time = 0.0;
  m_curves_next_iteration = 0;
  m_history_next_iteration = 0;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Informations de dépouillement en sortie de la boucle de calcul.
 *
 * Effectue une sortie pour le dépouillement si on est au temps final.
 * Cette sortie ne se fait qu'au temps final et garanti que le fichier
 * de dépouillement sera le même quel que soit le nombre de protections
 * reprises effectuées.
 */
void ArcanePostProcessingModule::
postProcessingExit()
{
  const Real current_time = subDomain()->commonVariables().globalTime();
  bool save_at_exit = false;
  if (subDomain()->timeLoopMng()->finalTimeReached())
    save_at_exit = options()->saveFinalTime();
  if (!save_at_exit)
    save_at_exit = options()->endExecutionOutput();
  if (save_at_exit){
    _saveAtTime(current_time);
  }

  // Affiche les statistiques d'exécutions
  Real total_time = m_post_processor_timer->totalTime();
  info() << "Total time for post-processing analysis output (second): " << total_time;
  Integer nb_time = m_post_processor_timer->nbActivated();
  if (nb_time!=0)
    info() << "Average time per output (second): " << total_time / nb_time
           << " (for " << nb_time << " outputs";
  IPostProcessorWriter* post_processor = options()->format();
  if (post_processor)
    post_processor->close();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie et écrit les valeurs pour le dépouillement.
 */
void ArcanePostProcessingModule::
exportData()
{
  if (m_is_output_at_current_iteration) {
    const Real current_time = subDomain()->commonVariables().globalTime();
    _saveAtTime(current_time);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Point d'entrée en début d'itération.
 */
void ArcanePostProcessingModule::
exportDataStart()
{
  ISubDomain* sd = subDomain();
  const CommonVariables& vc = sd->commonVariables();

  const Int32 global_iteration = vc.globalIteration();
  // Écrit les valeurs du dépouillement pour l'initialisation.
  if (global_iteration == 0)
    _saveAtTime(0.0);

  m_is_output_at_current_iteration = false;

  // Regarde s'il faut activer les historiques pour cette itération
  const Real old_time = vc.globalOldTime();
  const Real current_time = vc.globalTime();
  bool do_history_output = m_history_output_checker.check(old_time, current_time, global_iteration, 0);
  sd->timeHistoryMng()->setActive(do_history_output);

  // Regarde si des sorties de post-traitement sont prévues pour cette itération
  if (m_is_output_active) {
    bool do_at_current_iteration = m_output_checker.check(old_time, current_time, global_iteration, 0);
    if (do_at_current_iteration)
      _markCurrentIterationPostProcessing();
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcanePostProcessingModule::
_saveAtTime(Real saved_time)
{
  _resetCurrentIterationPostProcessing();

  const Int32 size = m_times.size();

  IVariableMng* vm = subDomain()->variableMng();

  // Ne sauvegarde pas si le temps actuel est le même que le précédent
  // (Sinon ca fait planter Ensight...)
  if (size!=0 && math::isEqual(m_times[size-1],saved_time))
    return;

  m_times.resize(size+1);
  m_times[size] = saved_time;

  _checkCreateOutputDir();

  if (m_is_output_active) {
    IPostProcessorWriter* post_processor = options()->format();
    post_processor->setBaseDirectoryName(m_output_directory.path());
    post_processor->setTimes(m_times);
    post_processor->setVariables(m_variables);
    post_processor->setGroups(m_groups);
    info() << " ";
    info() << "****  Output in progress at time " << saved_time <<"  ******";
  
    {
      Timer::Sentry ts(m_post_processor_timer);
      vm->writePostProcessing(post_processor);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Supprime les tags des variables post-processées lors de cette itération.
 */
void ArcanePostProcessingModule::
_resetCurrentIterationPostProcessing()
{
  m_is_output_at_current_iteration = false;
  for (VariableList::Enumerator v_iter(m_variables); ++v_iter;) {
    IVariable* v = *v_iter;
    v->removeTag(IVariable::TAG_POST_PROCESSING_AT_THIS_ITERATION);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Marque les variables comme étant post-processées lors de cette itération.
 */
void ArcanePostProcessingModule::
_markCurrentIterationPostProcessing()
{
  m_is_output_at_current_iteration = true;
  for (VariableList::Enumerator v_iter(m_variables); ++v_iter;) {
    IVariable* v = *v_iter;
    v->addTag(IVariable::TAG_POST_PROCESSING_AT_THIS_ITERATION, "1");
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
