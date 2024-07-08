// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableRef.cc                                              (C) 2000-2024 */
/*                                                                           */
/* Référence à une variable.                                                 */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArcanePrecomp.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/InvalidArgumentException.h"
#include "arcane/utils/IStackTraceService.h"
#include "arcane/utils/TraceInfo.h"
#include "arcane/utils/PlatformUtils.h"

#include "arcane/VariableRef.h"
#include "arcane/VariableBuildInfo.h"
#include "arcane/ISubDomain.h"
#include "arcane/IModule.h"
#include "arcane/IVariableMng.h"
#include "arcane/ArcaneException.h"

#include <set>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Classe interne pour gérer les fonctor appelés lors de la mise à jour
 * de la variable.
 *
 * Les fonctors sont en général ceux du wrapper C#. La principale difficulté
 * pour traiter ces fonctor est qu'ils sont gérer par le runtime C# et donc
 * utilisent un garbage collector. Il n'est donc pas possible de savoir
 * exactement quand ces fonctors seront détruits. Une instance de cette
 * classe ne doit donc pas être détruite explicitement. Lorsque la
 * variable possédant une instance de cette classe est détruite, elle
 * appelle destroy() pour signaler que l'objet peut être détruit. Dès
 * qu'il n'y a plus de fonctor référencés, cela signifie que tous les
 * objets C# sont détruits et donc on peut détruire l'instance.
 */
class VariableRef::UpdateNotifyFunctorList
{
 public:
  typedef void (*ChangedFunc)();
 public:
  UpdateNotifyFunctorList() : m_is_destroyed(false){}
 private:
  std::set<ChangedFunc> m_funcs;
  bool m_is_destroyed;
 public:
  void execute()
  {
    std::set<ChangedFunc>::const_iterator begin = m_funcs.begin();
    std::set<ChangedFunc>::const_iterator end = m_funcs.end();
    for( ; begin!=end; ++begin ){
      ChangedFunc f = *begin;
      (*f)();
    }
  }

  void destroy()
  {
    // Indique qu'on detruit.
    // Mais ne fais pas de delete tant que m_funcs n'est pas vide.
    m_is_destroyed = true;
    if (m_funcs.empty())
      delete this;
  }

  void add(ChangedFunc f)
  {
    m_funcs.insert(f);
  }

  void remove(ChangedFunc f)
  {
    m_funcs.erase(f);
    _checkDestroy();
  }
 public:
  static void* _add(VariableRef* var,void (*func)())
  {
    //std::cout << "_SET_MESH_VARIABLE_CHANGED_DELEGATE"
    //          << " name=" << var->name()
    //          << " func=" << (void*)func
    //          << " this=" << var
    //          << '\n';
    if (!var->m_notify_functor_list){
      var->m_notify_functor_list = new VariableRef::UpdateNotifyFunctorList();
    }
    var->m_notify_functor_list->add(func);
    return var->m_notify_functor_list;
  }
  static void _remove(UpdateNotifyFunctorList* functor_list,
                      void (*func)())
  {
    //std::cout << "_REMOVE_MESH_VARIABLE_CHANGED_DELEGATE"
    //          << " functor=" << functor_list
    //          << '\n';
    functor_list->remove(func);
  }
 private:

  void _checkDestroy()
  {
    if (m_is_destroyed && m_funcs.empty())
      delete this;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef::
VariableRef(const VariableBuildInfo& vbi)
: m_module(vbi.module())
, m_reference_property(vbi.property())
{
  //cout << "VAR NAME=" << vbi.name() << " this="
  //     << this << " module=" << m_module << '\n';
  _setAssignmentStackTrace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef::
VariableRef(IVariable* var)
: m_variable(var)
, m_reference_property(var->property())
{
  //cout << "VAR NAME=" << vbi.name() << " this="
  //     << this << " module=" << m_module << '\n';
  _setAssignmentStackTrace();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Constructeur pour une variable non enregistree.
 *
 * Ce constructeur n'est utilise que pour le wrapping C#. En C++,
 * il n'est pas accessible pour être sur que l'utilisateur n'a pas
 * de variables non référencées
 */
VariableRef::
VariableRef()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef::
VariableRef(const VariableRef& from)
: m_variable(from.m_variable)
, m_module(from.m_module)
, m_reference_property(from.m_reference_property)
{
  _setAssignmentStackTrace();
  //cout << "** TODO: check variable copy constructor with linked list\n";
  // NOTE:
  // C'est la variable qui met à jour m_previous_reference et m_next_reference
  // dans registerVariable.
  if (from.m_variable)
    registerVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef& VariableRef::
operator=(const VariableRef& rhs)
{
  if (this != &rhs) {
    if (rhs.m_variable != m_variable)
      _internalAssignVariable(rhs);
  }
  return (*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef::
~VariableRef()
{
  //cout << "DESTROY VAR NAME=" << name() << " this=" << this << '\n';
  //cout << "Unregistering variable ref " << this << '\n';
  //cout.flush();
  if (m_notify_functor_list)
    m_notify_functor_list->destroy();
  if (m_is_registered)
    unregisterVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
_setAssignmentStackTrace()
{
  m_assignment_stack_trace = String();
  if (hasTraceCreation()){
    IStackTraceService* stack_service = platform::getStackTraceService();
    if (stack_service){
      m_assignment_stack_trace = stack_service->stackTrace().toString();
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
unregisterVariable()
{
  _checkValid();
  m_variable->removeVariableRef(this);
  m_is_registered = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
registerVariable()
{
  _checkValid();
  m_variable->addVariableRef(this);
  m_is_registered = true;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
_internalInit(IVariable* variable)
{
  m_variable = variable;
  registerVariable();
  updateFromInternal();
  // Les variables autres que celles sur le maillage sont toujours utilisées
  // par défaut
  if (variable->itemKind()==IK_Unknown)
    setUsed(true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

eDataType VariableRef::
dataType() const 
{
  _checkValid();
  return m_variable->dataType();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
print(std::ostream& o) const 
{
  _checkValid();
  m_variable->print(o);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariableRef::
name() const
{
  _checkValid();
  return m_variable->name();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int VariableRef::
property() const
{
  _checkValid();
  return m_variable->property();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ISubDomain* VariableRef::
subDomain() const
{
  _checkValid();
  return m_variable->subDomain();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

IVariableMng* VariableRef::
variableMng() const
{
  _checkValid();
  return m_variable->variableMng();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer VariableRef::
checkIfSync(int max_print)
{
  _checkValid();
  return m_variable->checkIfSync(max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer VariableRef::
checkIfSameOnAllReplica(int max_print)
{
  _checkValid();
  return m_variable->checkIfSameOnAllReplica(max_print);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

int VariableRef::
referenceProperty() const
{
  return m_reference_property;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
updateFromInternal()
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
_throwInvalid() const
{
  String msg("Using a reference on a uninitialized variable");
  throw InternalErrorException(A_FUNCINFO,msg);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
setProperty(int property)
{
  if (!_checkValidPropertyChanged(property))
    throw InvalidArgumentException(A_FUNCINFO,"property",property);
  m_reference_property |= property;
  m_variable->notifyReferencePropertyChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
unsetProperty(int property)
{
  if (!_checkValidPropertyChanged(property))
    throw InvalidArgumentException(A_FUNCINFO,"property",property);
  m_reference_property &= ~property;
  m_variable->notifyReferencePropertyChanged();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Regarde si une propriété peut-être changée dynamiquement.
 */
bool VariableRef::
_checkValidPropertyChanged(int property)
{
  switch(property){
  case IVariable::PNoDump:
  case IVariable::PNoNeedSync:
  case IVariable::PSubDomainDepend:
  case IVariable::PSubDomainPrivate:
  case IVariable::PExecutionDepend:
  case IVariable::PNoRestore:
  case IVariable::PNoExchange:
  case IVariable::PPersistant:
  case IVariable::PNoReplicaSync:
    return true;
  case IVariable::PHasTrace:
  case IVariable::PPrivate:
    return false;
  default:
    break;
  }
  return false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
update()
{
  m_variable->update();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
setUpToDate()
{
  m_variable->setUpToDate();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 VariableRef::
modifiedTime()
{
  return m_variable->modifiedTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
addDependCurrentTime(const VariableRef& var)
{
  m_variable->addDepend(var.variable(),IVariable::DPT_CurrentTime);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
addDependPreviousTime(const VariableRef& var)
{
  m_variable->addDepend(var.variable(),IVariable::DPT_PreviousTime);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
addDependCurrentTime(const VariableRef& var,const TraceInfo& tinfo)
{
  m_variable->addDepend(var.variable(),IVariable::DPT_CurrentTime,tinfo);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
addDependPreviousTime(const VariableRef& var,const TraceInfo& tinfo)
{
  m_variable->addDepend(var.variable(),IVariable::DPT_PreviousTime,tinfo);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
removeDepend(const VariableRef& var)
{
  m_variable->removeDepend(var.variable());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
_setComputeFunction(IVariableComputeFunction* v)
{
  m_variable->setComputeFunction(v);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
addTag(const String& tagname,const String& tagvalue)
{
  m_variable->addTag(tagname,tagvalue);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
removeTag(const String& tagname)
{
  m_variable->removeTag(tagname);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VariableRef::
hasTag(const String& tagname) const
{
  return m_variable->hasTag(tagname);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

String VariableRef::
tagValue(const String& tagname) const
{
  return m_variable->tagValue(tagname);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef* VariableRef::
previousReference()
{
  return m_previous_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableRef* VariableRef::
nextReference()
{
  return m_next_reference;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
setPreviousReference(VariableRef* v)
{
  m_previous_reference = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
setNextReference(VariableRef* v)
{
  m_next_reference = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VariableRef::m_static_has_trace_creation = false;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
setTraceCreation(bool v)
{
  m_static_has_trace_creation = v;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

bool VariableRef::
hasTraceCreation()
{
  return m_static_has_trace_creation;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
_executeUpdateFunctors()
{
  if (m_notify_functor_list)
    m_notify_functor_list->execute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableRef::
_internalAssignVariable(const VariableRef& var)
{
  _setAssignmentStackTrace();
  if (m_is_registered)
    unregisterVariable();
  m_module = var.m_module;
  m_variable = var.m_variable;
  m_reference_property = var.m_reference_property;
  m_has_trace = false;
  // NE PAS TOUCHER A: m_notify_functor_list
  //NOTE:
  // C'est la variable qui met à jour m_previous_reference et m_next_reference
  // dans registerVariable.
  //cout << "** TODO: check variable operator= with linked list\n";
  registerVariable();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*
 * \internal
 * \brief Ajoute un fonctor pour le wrapping C#.
 */ 
extern "C" ARCANE_CORE_EXPORT void*
_AddVariableChangedDelegate(VariableRef* var,void (*func)())
{
  return VariableRef::UpdateNotifyFunctorList::_add(var,func);
}

/*
 * \internal
 * \brief Supprimer un fonctor pour le wrapping C#.
 */ 
extern "C" ARCANE_CORE_EXPORT void
_RemoveVariableChangedDelegate(VariableRef::UpdateNotifyFunctorList* functor_list,
                               void (*func)())
{
  VariableRef::UpdateNotifyFunctorList::_remove(functor_list,func);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
