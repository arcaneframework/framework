// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ArcanePostProcessingModule.cc                               (C) 2000-2009 */
/*                                                                           */
/* Module de post-traitement.                                                */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/


#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/Ptr.h"
#include "arcane/utils/List.h"

#include "arcane/EntryPoint.h"
#include "arcane/ISubDomain.h"
#include "arcane/IVariableMng.h"
#include "arcane/IApplication.h"
#include "arcane/IParallelMng.h"
#include "arcane/ItemGroup.h"
#include "arcane/Directory.h"
#include "arcane/ITimeHistoryMng.h"
#include "arcane/ServiceUtils.h"
#include "arcane/IPostProcessorWriter.h"
#include "arcane/SimpleProperty.h"
#include "arcane/MeshAccessor.h"
#include "arcane/IMesh.h"
#include "arcane/VariableTypes.h"
#include "arcane/CommonVariables.h"
#include "arcane/MathUtils.h"
#include "arcane/ITimeLoopMng.h"
#include "arcane/ItemEnumerator.h"
#include "arcane/ModuleFactory.h"
#include "arcane/Timer.h"
#include "arcane/IVariableAccessor.h"
#include "arcane/VariableCollection.h"

#include "arcane/OutputChecker.h"

#include "arcane/std/ArcanePostProcessing_axl.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Module de sortie pour le dépouillement.
 *
 Lorsque ce module est connecté, ce module gère les sorties pour
 le dépouillement.

 Si aucune variable n'est spécifiée, aucune sortie n'est effectuée. Le
 champs #m_do_output est alors à faux.
*/
class ArcanePostProcessingModule
: public ArcaneArcanePostProcessingObject
{
public:

  ArcanePostProcessingModule(const ModuleBuilder& cb);
  ~ArcanePostProcessingModule();

public:

  virtual VersionInfo versionInfo() const { return VersionInfo(0,1,2); }

public:

  virtual void exportData();
  virtual void exportDataStart();

  virtual void postProcessingStartInit();
  virtual void postProcessingInit();
  virtual void postProcessingExit();
  

private:

  OutputChecker m_output_checker;
  OutputChecker m_history_output_checker;
  VariableArrayReal m_times; //!< Instants de temps des sauvegardes
  bool m_do_output; //!< \a true si les sorties sont actives
  Directory m_output_directory; //!< Répertoire de sortie
  bool m_output_dir_created; //!< \a true si répertoire créé.
  VariableList m_variables;    //!< Liste des variables a exporter
  ItemGroupList m_groups; //!< Liste des groupes à exporter
  Timer* m_post_processor_timer; //!< Timer pour le temps passé à écrire

private:

  void _readConfig();
  void _saveAtTime(Real);

  void _checkCreateOutputDir();
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_REGISTER_MODULE_ARCANEPOSTPROCESSING(ArcanePostProcessingModule);

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ArcanePostProcessingModule::
ArcanePostProcessingModule(const ModuleBuildInfo& mbi)
: ArcaneArcanePostProcessingObject(mbi)
, m_output_checker(mbi.m_sub_domain,"PostProcessing")
, m_history_output_checker(mbi.m_sub_domain,"PostProcessingHistory")
, m_times(VariableBuilder(this,"ExportTimes"))
, m_do_output(true)
, m_output_dir_created(false)
, m_post_processor_timer(0)
{
  m_output_checker.assignIteration(&m_next_iteration,&options()->outputPeriod);
  m_output_checker.assignGlobalTime(&m_next_global_time,&options()->outputFrequency);

  m_history_output_checker.assignIteration(&m_history_next_iteration,&options()->outputHistoryPeriod);
  m_post_processor_timer = new Timer(mbi.m_sub_domain,"PostProcessorTimer",Timer::TimerReal);
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
        fatal() << "PostTreatment: no variable with name '" << varname
                << "' exists";
      eItemKind ik = var->itemKind();
      if (ik!=IK_Node && ik!=IK_Edge && ik!=IK_Face && ik!=IK_Cell)
        fatal() << "PostTreatment: variable (" << varname << ") must"
                << " be a mesh variable (node, edge, face or cell)";
      if (used_variables.find(varname)==used_variables.end()){
        info() << "Variable <" << varname << ">";
        m_variables.add(var);
        used_variables.insert(varname);
        var->addTag("PostProcessing","1");
      }
      else{
        warning() << "Variable <" << varname << "> required twice during post-processing analysis";
      }
    }
  }
  else
    m_do_output = false;

  if (nb_group!=0){
    std::set<String> used_groups; // Liste des groupes déjà indiquées
    Integer index = 0;
    //m_group_list.resize(nb_group);
    info() << " ";
    info() << "-- List of output groups (" << nb_group << " groups):";
    for( Integer i=0; i<nb_group; ++i ){
      ItemGroup group = options()->output().group[i];
      if (group.null())
        continue;
      String groupname = group.name();
      if (used_groups.find(groupname)==used_groups.end()){
        ++index;
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
	if(mesh->isAmrActivated())
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
  if (options()->outputHistoryShrink==true){
    //info()<< "\33[42;30m" << "postProcessingStartInit Setting History to be shrank!" << "\33[m";
    subDomain()->timeHistoryMng()->setShrinkActive(options()->outputHistoryShrink);
  }else{
    //info()<< "\33[42;30m" << "postProcessingStartInit Plain History!" << "\33[m";
  }

  // initialize parameter with a dry call to checker
  const CommonVariables& vc = subDomain()->commonVariables();
  Real old_time = vc.globalOldTime();
  Real current_time = vc.globalTime();
  /* bool do_output = */ m_output_checker.check(old_time,current_time,
                                                vc.globalIteration(),0);
  /* bool do_output = */ m_history_output_checker.check(old_time,current_time,
                                                        vc.globalIteration(),0);
  if (options()->saveInit()){
    _saveAtTime(current_time);
  }
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
  Real current_time = subDomain()->commonVariables().globalTime();
  bool save_at_exit = false;
  if (subDomain()->timeLoopMng()->finalTimeReached())
    save_at_exit = options()->saveFinalTime();
  if (!save_at_exit)
    save_at_exit = options()->endExecutionOutput();
  if (save_at_exit){
    _saveAtTime(current_time);
  }

  // Affiche statistiques d'exécutions
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
  const CommonVariables& vc = subDomain()->commonVariables();
  Real old_time = vc.globalOldTime();
  Real current_time = vc.globalTime();
  bool do_output = m_output_checker.check(old_time,current_time,
                                          vc.globalIteration(),0);
  if (do_output)
    _saveAtTime(current_time);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Point d'entrée en début d'itération.
 */
void ArcanePostProcessingModule::
exportDataStart()
{
  // Ecrit les valeurs du dépouillement pour l'initialisation.
  if (subDomain()->commonVariables().globalIteration()==0)
    _saveAtTime(0.);

  // Regarde s'il faut activer les historiques pour cette itération
  const CommonVariables& vc = subDomain()->commonVariables();
  Real old_time = vc.globalOldTime();
  Real current_time = vc.globalTime();
  bool do_history_output = m_history_output_checker.check(old_time,current_time,
                                                          vc.globalIteration(),0);

  subDomain()->timeHistoryMng()->setActive(do_history_output);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ArcanePostProcessingModule::
_saveAtTime(Real saved_time)
{
  Integer size = m_times.size();

  IVariableMng* vm = subDomain()->variableMng();
  //IParallelMng* pm = subDomain()->parallelMng();

  // Ne sauvegarde pas si le temps actuel est le même que le précédent
  // (Sinon ca fait planter Ensight...)
  if (size!=0 && math::isEqual(m_times[size-1],saved_time))
    return;

  m_times.resize(size+1);
  m_times[size] = saved_time;

  _checkCreateOutputDir();

  if (m_do_output){
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

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
