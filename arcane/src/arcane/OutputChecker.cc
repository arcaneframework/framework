// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* OutputChecker.cc                                            (C) 2000-2016 */
/*                                                                           */
/* Sorties basées sur un temps (physique ou CPU) ou un nombre d'itération.   */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"

#include "arcane/utils/CheckedConvert.h"
#include "arcane/utils/ITraceMng.h"

#include "arcane/ISubDomain.h"
#include "arcane/ICaseFunction.h"
#include "arcane/CommonVariables.h"

#include "arcane/OutputChecker.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

OutputChecker::
OutputChecker(ISubDomain* sd,const String& name)
: m_sub_domain(sd)
, m_name(name)
, m_out_type(OutTypeNone)
, m_next_iteration(0)
, m_next_global_time(0)
, m_next_cpu_time(0)
, m_step_iteration(0)
, m_step_global_time(0)
, m_step_cpu_time(0)
{}

void OutputChecker::
assignGlobalTime(VariableScalarReal* variable,const CaseOptionReal* option)
{
  m_next_global_time = variable;
  m_step_global_time = option;
}

void OutputChecker::
assignCPUTime(VariableScalarInteger* variable,const CaseOptionInteger* option)
{
  m_next_cpu_time = variable;
  m_step_cpu_time = option;
}

void OutputChecker::
assignIteration(VariableScalarInteger* variable,const CaseOptionInteger* option)
{
  m_next_iteration = variable;
  m_step_iteration = option;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vérifie s'il faut effectuer une sortie.
 *
 * \arg \a old_time et \a current_time sont utilisés pour les sorties
 * en temps physique.
 * \arg \a current_iteration est utilisé pour les sorties en nombre d'itérations
 * \arg \a cpu_time_used est utilisé pour les sorties en temps CPU.

 Lorsqu'un type de sortie n'est pas disponible, la ou les valeurs associées
 ne sont pas utilisées et peuvent être quelconques.

 \param old_time temps physique de l'itération précédente.
 \param current_time temps physique courant.
 \param current_iteration itération courante.
 \param cpu_time_used temps cpu utilisé
 */
bool OutputChecker::
check(Real old_time,Real current_time,
      Integer current_iteration,Integer cpu_time_used,
      const String& from_function)
{
  String function_id = "OutputChecker::check >>> ";

  if (m_out_type==OutTypeNone)
    return false;

  ITraceMng* trace = m_sub_domain->traceMng();

  // \a true si on va effectuer une sortie
  bool do_output = false;
  switch(m_out_type){
  case OutTypeGlobalTime:
    {
      Real next_time = (*m_next_global_time)();
      trace->debug()<<from_function<<function_id << "Next out: " << next_time << " Saved Time: " << old_time
                    << " Current time: " << current_time << " step=" << m_step_global_time;
      //if (math::isEqual(m_next_time(),old_time))
      if (math::isEqual(next_time,current_time))
        do_output = true;
      // Pour faire une sortie, il faut qu'on ait dépassé
      // le temps précédent et que le temps courant soit #strictement# supérieur
      // au temps de la sortie.
      else if (next_time>old_time && next_time<current_time){
        do_output = true;
      }
      // TODO: diviser par le nombre de pas de sorties
      if (do_output || next_time<old_time){
        //Real to_add = m_step_time(); //old_time);
        // S'il le temps est donné par une table de marche, il faut prendre
        // sa valeur à l'instant courant.
        //Il faut prendre la valeur 
        Real to_add = m_step_global_time->valueAtParameter(current_time); //old_time);
        if (!math::isZero(to_add)){
          //m_next_time += to_add;
          Real diff = (current_time-next_time) / to_add;
          if (diff<0.)
            *m_next_global_time = next_time+to_add;
          else{
            double i_diff = math::floor(diff);
            *m_next_global_time = next_time+(to_add*(i_diff+1));
          }
          trace->debug()<<from_function<<function_id
                        << "Next output at time " << next_time
                        << " (" << to_add << ' ' << old_time
                        << ' ' << current_iteration << ")";
        }
      }
    }
    break;
  case OutTypeIteration:
    {
      Integer next_iteration = (*m_next_iteration)();
      Integer to_add = m_step_iteration->valueAtParameter(current_time);

      if (next_iteration>current_iteration+to_add){
        // TH: En cas de reprise avec diminution de la période de sortie dans jdd
        // TH: on reprend les sorties tout de suite
        next_iteration = current_iteration;
      }

      if (next_iteration==current_iteration) do_output = true;
       
      // Calcule la prochaine itération de sauvegarde si nécessaire
      if (next_iteration<=current_iteration){
        if (to_add!=0){
          Integer diff = (current_iteration-next_iteration) / to_add;
          (*m_next_iteration) = next_iteration + ((diff+1)*to_add);
          trace->debug()<<from_function<<function_id
                        << "Next output at iteration " << (*m_next_iteration)()
                        << " (" << to_add << ' ' << old_time
                        << ' ' << current_iteration << " diff=" << diff << ")";
        }
      }
    }
    break;
  case OutTypeCPUTime:
    {
      Integer next_cpu_time = (*m_next_cpu_time)();
      Integer cpu_time = cpu_time_used;
      // Converti le temps CPU en minutes (à garder cohérent avec la conversion dans _recomputeTypeCPUTime())

      Integer current_cpu_time  = cpu_time / 60;       

      if (next_cpu_time<=current_cpu_time)
        do_output = true;
       
      // Calcule le prochain temps CPU de sauvegarde si nécessaire
      if (next_cpu_time<=current_cpu_time){
        Integer to_add = m_step_cpu_time->valueAtParameter(current_time);
        if (to_add!=0){
          Integer diff = (current_cpu_time-next_cpu_time) / to_add;
          (*m_next_cpu_time) = next_cpu_time + ((diff+1)*to_add);
          trace->debug()<<from_function<<function_id
                        << "Next output at cpu time " << (*m_next_cpu_time)()
                        << " (" << to_add << ' ' << current_cpu_time << " diff=" << diff << ")";
        }
      }

      // Ne fait rien la première minute.
      if (current_cpu_time==0)
        do_output = false;

      //if (do_output)
      //cerr << "** ** OUTPUT AT CPU TIME " << cpu_time
      //<< ' ' << current_cpu_time << ' ' << (*m_next_cpu_time)()
      //   << ' ' << (*m_step_cpu_time)() << '\n';

    }
    //msg->warning() << "Not implemented";
    break;
  case OutTypeNone:
    break;
  }

  return do_output;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OutputChecker::
_recomputeTypeGlobalTime()
{
  Real current_time = m_sub_domain->commonVariables().globalTime();
  Real step = m_step_global_time->valueAtParameter(current_time);

  Real old_next = (*m_next_global_time)();
  Real next = old_next;
  Real current_value = current_time;
  if (!math::isZero(step)){
    Real index = ::floor(current_value/step);
    next = step*(index+1.0);
    *m_next_global_time = next;
  }

  ITraceMng* tm = m_sub_domain->traceMng();
  tm->info(4) << "Recompute OutputChecker for Global Time: old=" << old_next
             << " new=" << next;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OutputChecker::
_recomputeTypeCPUTime()
{
  Real current_time = m_sub_domain->commonVariables().globalTime();
  Integer step = m_step_cpu_time->valueAtParameter(current_time);

  Integer old_next = (*m_next_cpu_time)();
  Integer next = old_next;
  double current_value = math::floor(m_sub_domain->commonVariables().globalCPUTime());

  // Converti en minute (à garder cohérent avec la conversion dans check())
  current_value /= 60.0;

  if (step!=0){
    Integer index = CheckedConvert::toInteger(current_value / step);
    next = step*(index+1);
    *m_next_cpu_time = next;
  }

  ITraceMng* tm = m_sub_domain->traceMng();
  tm->info(4) << "Recompute OutputChecker for CPU Time: old=" << old_next << " new=" << next;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OutputChecker::
_recomputeTypeIteration()
{
  Real current_time = m_sub_domain->commonVariables().globalTime();
  Integer step = m_step_iteration->valueAtParameter(current_time);

  Integer old_next = (*m_next_iteration)();
  Integer next = old_next;
  Integer current_value = m_sub_domain->commonVariables().globalIteration();
  if (step!=0){
    if (old_next<=current_value){
      *m_next_iteration = current_value;
    }
    else{
      Integer index = current_value / step;
      next = step*(index+1);
      *m_next_iteration = next;
    }
  }

  ITraceMng* tm = m_sub_domain->traceMng();
  tm->info(4) << "Recompute OutputChecker for CPU Time: old=" << old_next << " new=" << next;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OutputChecker::
initialize()
{
  initialize(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void OutputChecker::
initialize(bool recompute_next_value)
{
  ITraceMng* trace = m_sub_domain->traceMng();
  m_out_type = OutTypeNone;

  ICaseFunction* func = 0;
  // Prioritairement prend les sorties en temps physique, puis en
  // nombre d'itérations, puis en temps CPU.
  if (m_step_global_time && m_next_global_time && m_step_global_time->isPresent()){
    m_out_type = OutTypeGlobalTime;
    func = m_step_global_time->function();
    if (func)
      trace->info() << "Output in real time controlled by function '" << func->name() << "'.";
    else
      trace->info() << "Output in real time every " << (*m_step_global_time)() << " seconds.";
    if (recompute_next_value)
      _recomputeTypeGlobalTime();
  }
  else if (m_step_iteration && m_next_iteration && m_step_iteration->value()!=0){
    m_out_type = OutTypeIteration;
    func = m_step_iteration->function();
    if (func)
      trace->info() << "Output in iterations controlled by function '" << func->name() << "'.";
    else
      trace->info() << "Output every " << (*m_step_iteration)() << " iterations.";
    if (recompute_next_value)
      _recomputeTypeIteration();
  }
  else if (m_step_cpu_time && m_next_cpu_time && m_step_cpu_time->isPresent()){
    m_out_type = OutTypeCPUTime;
    func = m_step_cpu_time->function();
    if (func)
      trace->info() << "Output in CPU time controlled by function '" << func->name() << "'.";
    else
      trace->info() << "Output in CPU time every " << (*m_step_cpu_time)() << " minutes.";
    if (recompute_next_value)
      _recomputeTypeCPUTime();
  }
  else
    trace->info() << "No output required.";
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Real OutputChecker::
nextGlobalTime() const
{
  Real v = 0.0;
  if (m_next_global_time)
    v = (*m_next_global_time)();
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer OutputChecker::
nextIteration() const
{
  Integer v = 0;
  if (m_next_iteration)
    v = (*m_next_iteration)();
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer OutputChecker::
nextCPUTime() const
{
  Integer v = 0;
  if (m_next_cpu_time)
    v= (*m_next_cpu_time)();
  return v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
