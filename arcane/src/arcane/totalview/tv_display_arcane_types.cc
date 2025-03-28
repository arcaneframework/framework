// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* tv_display_arcane_types.cc                                  (C) 2000-2025 */
/*                                                                           */
/* Informations pour le debugging avec totalview                             */
/*---------------------------------------------------------------------------*/

#include "arcane/utils/String.h"
#include "arcane/utils/Array.h"
#include "arcane/utils/Real3.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/Item.h"
#include "arcane/ItemInternal.h"
#include "arcane/ItemGroup.h"
#include "arcane/ItemGroupImpl.h"
#include "arcane/IMesh.h"
#include "arcane/IItemFamily.h"
#include "arcane/MeshVariable.h"
#include "arcane/Variable.h"
#include "arcane/VariableRefScalar.h"

#include "arcane/totalview/tv_data_display.h"

/*---------------------------------------------------------------------------*/
// TODO: Ne pas toujours afficher le arcane_ttf_header
// TODO: Supprimer _ArrayStruct et utiliser une chaine de caractères pour
// afficher les infos des tableaux.

// Cette fonctionnalité est accessible avec Totalview >= 8.9
// La doc est disponible ici:
// http://www.roguewave.com/support/product-documentation/totalview-family.aspx
// Les sections de code ici présentes ne sont pas un exemple à suivre en dehors
// du contexte très particulier de ttf.
// En particulier la technique de __MY_ItemEnumerator permet d'accéder à des
// champs privés de ItemEnumerator

// Un point important est que le 3ème argument de TV_ttf_add_row() doit
// contenir un pointeur sur la valeur à afficher et que ce pointeur doit
// rester valide après l'appel. Il n'est donc pas possible d'utiliser des
// variables locales.

#ifdef __GCC__
#define ATTR_USED __attribute__((used))
#else
#define ATTR_USED
#endif

namespace Arcane
{
class TotalviewAdapter
{
 public:
  static ItemSharedInfo* getItemSharedInfo(const ItemInternal* v)
  {
    return ItemInternalCompatibility::_getSharedInfo(v);
  }
  static ItemInternal* getInternal(const Item* v)
  {
    return ItemCompatibility::_itemInternal(*v);
  }
};
}

namespace
{
// Pour savoir si on affiche un message dans totalview indiquant qu'on utilise
// les TTF.
bool global_print_ttf = false;

void
arcane_ttf_header()
{
  if (global_print_ttf)
    TV_ttf_add_row("__Warning__",TV_ttf_type_ascii_string,"ttf enabled (version=2)!");
}
}

static inline void
show_ttf_internal_flag(Arcane::Integer flags,
                       Arcane::Integer flag,
                       const char * name)
{  
  if (flags & flag){
    char strtype[256];
    snprintf(strtype,sizeof(strtype),"flag=%s",name);
    TV_ttf_add_row(strtype,TV_ttf_type_ascii_string,"on");
  }
}
namespace
{
// A mettre en correspondance avec Arccore::ArrayMetaData
class _ArrayMetaData
{
 public:
  //! Nombre de références sur cet objet.
  Arccore::Int64 nb_ref;
  //! Nombre d'éléments alloués
  Arccore::Int64 capacity;
  //! Nombre d'éléments du tableau (pour les tableaux 1D)
  Arccore::Int64 size;
  //! Taille de la première dimension (pour les tableaux 2D)
  Arccore::Int64 dim1_size;
  //! Taille de la deuxième dimension (pour les tableaux 2D)
  Arccore::Int64 dim2_size;
};
// ATTENTION: classe qui mime la représentation de Arccore::AbstractArray.
// A modifier si cette dernière évolue.
class _ArrayStruct
{
 public:
  virtual ~_ArrayStruct(){}
  void* m_p;
  _ArrayMetaData* m_md;
};
// ATTENTION: classe qui mime la représentation de Arcane::ItemSharedInfo.
// A modifier si cette dernière évolue.
class _ItemSharedInfo
{
 public:
  Arcane::MeshItemInternalList* m_items = nullptr;
  Arcane::ItemInternalConnectivityList* m_connectivity;
  Arcane::IItemFamily* m_item_family = nullptr;
  Arcane::ItemTypeMng* m_item_type_mng = nullptr;
  Arcane::Int64ArrayView m_unique_ids;
  Arcane::Int32ArrayView m_parent_item_ids;
  Arcane::Int32ArrayView m_owners;
  Arcane::Int32ArrayView m_flags;
  Arcane::Int16ArrayView m_type_ids;
  Arcane::eItemKind m_item_kind;
  Arcane::Int32 m_nb_parent;
  Arcane::ConstArrayView<Arcane::ItemInternal*> m_items_internal;
};
}
/*---------------------------------------------------------------------------*/
/*                             Arcane::Array  (sur type de base)             */
/*---------------------------------------------------------------------------*/

namespace Arcane
{
template<typename type> int
_displayArray(const AbstractArray<type>* obj,const char* type_name)
{
  Arccore::Span<const type> view = *obj;
  const _ArrayStruct* true_ptr = reinterpret_cast<const _ArrayStruct*>(obj);
  _ArrayMetaData* true_obj = true_ptr->m_md;
  char strtype[1024];
  arcane_ttf_header();
  TV_ttf_add_row("size","Arcane::Int64",&true_obj->size);
  TV_ttf_add_row("capacity","Arcane::Int64",&true_obj->capacity);
  snprintf(strtype,sizeof(strtype),"%s[%ld]",type_name,view.size());
  TV_ttf_add_row("data",strtype,view.data());
  return TV_ttf_format_ok;
}
}

#define TV_DISPLAY_ARRAY_TYPE(type)\
ATTR_USED int \
TV_ttf_display_type(const Arcane::Array<type>* obj)\
{ return _displayArray(obj,#type); }\
ATTR_USED int \
TV_ttf_display_type(const Arcane::UniqueArray<type>* obj)\
{ return _displayArray(obj,#type); }\
ATTR_USED int \
TV_ttf_display_type(const Arcane::SharedArray<type>* obj)\
{ return _displayArray(obj,#type); }

// Les instances souhaitées doivent être explicitement instanciées
TV_DISPLAY_ARRAY_TYPE(bool)
TV_DISPLAY_ARRAY_TYPE(int)
TV_DISPLAY_ARRAY_TYPE(double)
TV_DISPLAY_ARRAY_TYPE(unsigned)
TV_DISPLAY_ARRAY_TYPE(long)
TV_DISPLAY_ARRAY_TYPE(unsigned long)
TV_DISPLAY_ARRAY_TYPE(Arcane::Real3)
TV_DISPLAY_ARRAY_TYPE(Arcane::Item)
TV_DISPLAY_ARRAY_TYPE(Arcane::Node)
TV_DISPLAY_ARRAY_TYPE(Arcane::Edge)
TV_DISPLAY_ARRAY_TYPE(Arcane::Face)
TV_DISPLAY_ARRAY_TYPE(Arcane::Cell)

#undef TV_DISPLAY_ARRAY_TYPE

/*---------------------------------------------------------------------------*/
/*             Arcane::ArrayView et Arcane::ConstArrayView                   */
/*---------------------------------------------------------------------------*/

#define TV_DISPLAY_ARRAY_TYPE(type)                                     \
ATTR_USED                                                               \
int                                                                     \
TV_ttf_display_type(const Arcane::ConstArrayView<type> * obj)           \
{                                                                       \
  char strtype[1024];                                                   \
  snprintf(strtype,sizeof(strtype),"%s[%d]",#type,obj->size());         \
  TV_ttf_add_row("data",strtype,obj->data());           \
  return TV_ttf_format_ok;                                              \
}                                                                       \
ATTR_USED                                                               \
int                                                                     \
TV_ttf_display_type(const Arcane::ArrayView<type> * obj)                \
{                                                                       \
  char strtype[1024];                                                   \
  snprintf(strtype,sizeof(strtype),"%s[%d]",#type,obj->size());         \
  TV_ttf_add_row("data",strtype,obj->data());           \
  return TV_ttf_format_ok;                                              \
}

// Les instances souhaitées doivent être explicitement instanciées
TV_DISPLAY_ARRAY_TYPE(bool)
TV_DISPLAY_ARRAY_TYPE(int)
TV_DISPLAY_ARRAY_TYPE(double)
TV_DISPLAY_ARRAY_TYPE(unsigned)
TV_DISPLAY_ARRAY_TYPE(long)
TV_DISPLAY_ARRAY_TYPE(unsigned long)
TV_DISPLAY_ARRAY_TYPE(Arcane::Real3)
TV_DISPLAY_ARRAY_TYPE(Arcane::Item)
TV_DISPLAY_ARRAY_TYPE(Arcane::Node)
TV_DISPLAY_ARRAY_TYPE(Arcane::Edge)
TV_DISPLAY_ARRAY_TYPE(Arcane::Face)
TV_DISPLAY_ARRAY_TYPE(Arcane::Cell)

#undef TV_DISPLAY_ARRAY_TYPE

/*---------------------------------------------------------------------------*/
/*                         Arcane::ItemInternal                              */
/*---------------------------------------------------------------------------*/

ATTR_USED
int
TV_ttf_display_type(const Arcane::ItemInternal * obj)
{
  // ATTENTION: cette implémentation aggresse directement les structures Arcane.
  // En cas de changement d'implémentation d'Arcane des incompatibilités peuvent
  // se produire

  using Arcane::Integer;
  char strtype[32];

  arcane_ttf_header();
  TV_ttf_add_row("local_id","Integer",reinterpret_cast<const int*>(obj));
  _ItemSharedInfo* shared_info = reinterpret_cast<_ItemSharedInfo*>(Arcane::TotalviewAdapter::getItemSharedInfo(obj));
  const Integer local_id = obj->localId();

  TV_ttf_add_row("unique_id","Int64",shared_info->m_unique_ids.ptrAt(local_id));
  TV_ttf_add_row("kind","eItemKind",&shared_info->m_item_kind);
  
  snprintf(strtype,sizeof(strtype),"type=%s",obj->typeInfo()->typeName().localstr());
  //TV_ttf_add_row(strtype,"ItemTypeInfo*",&shared_info->m_item_type);

  {
    // Nodes
    for(Integer i=0;i<obj->nbNode();++i){
      Arcane::ItemInternal * item = obj->internalNode(i);
      snprintf(strtype,sizeof(strtype),"node[%d] (lid=%d uid=%ld)",i,item->localId(),item->uniqueId().asInt64());
      TV_ttf_add_row(strtype,"ItemInternal*",item);
    }
  }
  {
    // Edges
    for(Integer i=0;i<obj->nbEdge();++i){
      Arcane::ItemInternal* item = obj->internalEdge(i);
      snprintf(strtype,sizeof(strtype),"edge[%d] (lid=%d uid=%ld)",i,item->localId(),item->uniqueId().asInt64());
      TV_ttf_add_row(strtype,"ItemInternal*",item);
    }
  }
  {
    // Faces
      for(Integer i=0;i<obj->nbFace();++i) {
        Arcane::ItemInternal * item = obj->internalFace(i);
        snprintf(strtype,sizeof(strtype),"face[%d] (lid=%d uid=%ld)",i,item->localId(),item->uniqueId().asInt64());
        TV_ttf_add_row(strtype,"ItemInternal*",item);
      }
  }
  {
    // Cells
    for(Integer i=0;i<obj->nbCell();++i){
      Arcane::ItemInternal * item = obj->internalCell(i);
      snprintf(strtype,sizeof(strtype),"cell[%d] (lid=%d uid=%ld)",i,item->localId(),item->uniqueId().asInt64());
      TV_ttf_add_row(strtype,"ItemInternal*",item);
    }
  }

  // TODO pour les DualNode et Link, mais ce n'est pas si simple...

  const Integer flags = obj->flags();
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_Boundary,"Boundary");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_HasFrontCell,"HasFrontCell");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_HasBackCell,"HasBackCell");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_FrontCellIsFirst,"FrontCellIsFirst");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_BackCellIsFirst,"BackCellIsFirst");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_Own,"Own");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_Added,"Added");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_Suppressed,"Suppressed");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_Shared,"Shared");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_SubDomainBoundary,"SubDomainBoundary");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_JustAdded,"JustAdded");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_NeedRemove,"NeedRemove");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_SlaveFace,"SlaveFace");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_MasterFace,"MasterFace");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_Detached,"Detached");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_HasTrace,"HasTrace");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_UserMark1,"UserMark1");
  show_ttf_internal_flag(flags,Arcane::ItemInternal::II_UserMark2,"UserMark2");

  return TV_ttf_format_ok;
}

/*---------------------------------------------------------------------------*/
/*                         Arcane::String                                    */
/*---------------------------------------------------------------------------*/

int
TV_ttf_display_type(const Arcane::String * obj)
{
  //arcane_ttf_header();
  TV_ttf_add_row("data",TV_ttf_type_ascii_string,obj->localstr());
  //GG: avec elide, on evite une indentation supplementaire
  return TV_ttf_format_ok_elide;
}

/*---------------------------------------------------------------------------*/
/*                         Arcane::IItemFamily                               */
/*---------------------------------------------------------------------------*/

ATTR_USED
int
TV_ttf_display_type(const Arcane::IItemFamily * obj)
{
  arcane_ttf_header();
  char strtype[1024];
  Arcane::eItemKind kind = obj->itemKind();
  TV_ttf_add_row("kind",TV_ttf_type_ascii_string,Arcane::itemKindName(kind));
  TV_ttf_add_row("name",TV_ttf_type_ascii_string,obj->name().localstr());
  TV_ttf_add_row("fullname",TV_ttf_type_ascii_string,obj->fullName().localstr());
  snprintf(strtype,sizeof(strtype),"size=%d",obj->nbItem());
  TV_ttf_add_row(strtype,TV_ttf_type_ascii_string,"");
  snprintf(strtype,sizeof(strtype),"maxLocalId=%d",obj->maxLocalId());
  TV_ttf_add_row(strtype,TV_ttf_type_ascii_string,"");
  Arcane::ConstArrayView<Arcane::ItemInternal*> view = const_cast<Arcane::IItemFamily *>(obj)->itemsInternal();
  snprintf(strtype,sizeof(strtype),"Arcane::ItemInternal*[%d]",view.size());
  TV_ttf_add_row("itemsInternal",strtype,view.data());
  return TV_ttf_format_ok;
}

/*---------------------------------------------------------------------------*/
/*                         Arcane::ItemEnumerator                            */
/*---------------------------------------------------------------------------*/

// Structure symétrique pour permettre des accès à des données privées. 
// Il faut etre certain que cette structure soit identique a ItemEnumerator
struct __MY_ItemEnumerator
{
  Arcane::ItemInternal** m_items;
  const Arcane::Int32* ARCANE_RESTRICT m_local_ids;
  Arcane::Integer m_index;
  Arcane::Integer m_count;
  const Arcane::ItemGroupImpl * m_group_impl; // pourrait être retiré en mode release si nécessaire
};

int
TV_ttf_display_type(const Arcane::ItemEnumerator* obj)
{
  arcane_ttf_header();

  __MY_ItemEnumerator* mobj = (__MY_ItemEnumerator*)obj;

  Arcane::Integer index = mobj->m_index;

  TV_ttf_add_row("index",TV_ttf_type_int,&mobj->m_index);
  TV_ttf_add_row("count",TV_ttf_type_int,&mobj->m_count);
  TV_ttf_add_row("local_id",TV_ttf_type_int,&mobj->m_local_ids[index]);
  if (index<mobj->m_count)
    TV_ttf_add_row("item","Arcane::ItemInternal",(mobj->m_items[mobj->m_local_ids[index]]));
  return TV_ttf_format_ok;
}

/*---------------------------------------------------------------------------*/
/*                         Arcane::Item                                      */
/*---------------------------------------------------------------------------*/

int
TV_ttf_display_type(const Arcane::Item* obj)
{
  TV_ttf_add_row("item","Arcane::ItemInternal",Arcane::TotalviewAdapter::getInternal(obj));
  return TV_ttf_format_ok_elide;
}

/*---------------------------------------------------------------------------*/
/*                         Arcane::ItemGroup                                 */
/*---------------------------------------------------------------------------*/

int
TV_ttf_display_type(const Arcane::ItemGroup* obj)
{
  arcane_ttf_header();
  char strtype[1024];
  TV_ttf_add_row("name",TV_ttf_type_ascii_string,obj->name().localstr());
  TV_ttf_add_row("fullName",TV_ttf_type_ascii_string,obj->fullName().localstr());
  TV_ttf_add_row("family","Arcane::IItemFamily", obj->itemFamily());
  TV_ttf_add_row("internal","Arcane::ItemGroupImpl", obj->internal());
  Arcane::ConstArrayView<Arcane::Int32> view = obj->view().localIds();
  snprintf(strtype,sizeof(strtype),"Arcane::Int32[%d]",view.size());
  TV_ttf_add_row("localIds",strtype,view.unguardedBasePointer());
  return TV_ttf_format_ok;
}

/*---------------------------------------------------------------------------*/
/*                         Arcane::IMesh                                     */
/*---------------------------------------------------------------------------*/

int
TV_ttf_display_type(const Arcane::IMesh* cobj)
{
  Arcane::IMesh * obj = const_cast<Arcane::IMesh*>(cobj);
  arcane_ttf_header();
  TV_ttf_add_row("name",TV_ttf_type_ascii_string,obj->name().localstr());
  TV_ttf_add_row("cellfamily","Arcane::IItemFamily", obj->cellFamily());
  TV_ttf_add_row("facefamily","Arcane::IItemFamily", obj->faceFamily());
  TV_ttf_add_row("edgefamily","Arcane::IItemFamily", obj->edgeFamily());
  TV_ttf_add_row("nodefamily","Arcane::IItemFamily", obj->nodeFamily());
  return TV_ttf_format_ok;
}

/*---------------------------------------------------------------------------*/
/*                         Arcane::VariableItem*                             */
/*---------------------------------------------------------------------------*/

void
_displayVariable(const Arcane::IVariable* var)
{
  TV_ttf_add_row("name",TV_ttf_type_ascii_string,var->name().localstr());
  TV_ttf_add_row("dataType",TV_ttf_type_ascii_string,Arcane::dataTypeName(var->dataType()));
  int dimension = var->dimension();
  TV_ttf_add_row("dimension",TV_ttf_type_int,&dimension);
  TV_ttf_add_row("itemKind",TV_ttf_type_ascii_string,Arcane::itemKindName(var->itemKind()));

  TV_ttf_add_row("groupName",TV_ttf_type_ascii_string,var->itemGroupName().localstr());
  TV_ttf_add_row("group","Arcane::ItemGroupImpl*", var->itemGroup().internal());
  TV_ttf_add_row("itemFamilyName",TV_ttf_type_ascii_string,var->itemFamilyName().localstr());
  TV_ttf_add_row("family","Arcane::IItemFamily", var->itemFamily());
  TV_ttf_add_row("meshName",TV_ttf_type_ascii_string,var->meshName().localstr());

  int nbReference = var->nbReference();
  TV_ttf_add_row("nbReference",TV_ttf_type_int,&nbReference);

  int properties = var->property();
  show_ttf_internal_flag(properties, Arcane::IVariable::PNoDump, "NoDump");
  show_ttf_internal_flag(properties, Arcane::IVariable::PNoNeedSync, "NoNeedSync");
  show_ttf_internal_flag(properties, Arcane::IVariable::PHasTrace, "HasTrace");
  show_ttf_internal_flag(properties, Arcane::IVariable::PSubDomainDepend, "SubDomainDepend");
  show_ttf_internal_flag(properties, Arcane::IVariable::PSubDomainPrivate, "SubDomainPrivate");
  show_ttf_internal_flag(properties, Arcane::IVariable::PExecutionDepend, "ExecutionDepend");
  show_ttf_internal_flag(properties, Arcane::IVariable::PPersistant, "Persistant");
  show_ttf_internal_flag(properties, Arcane::IVariable::PPrivate, "Private");
  show_ttf_internal_flag(properties, Arcane::IVariable::PNoRestore, "NoRestore");
  show_ttf_internal_flag(properties, Arcane::IVariable::PNoExchange, "NoExchange");
  show_ttf_internal_flag(properties, Arcane::IVariable::PTemporary, "Temporary");
}

ATTR_USED int
TV_ttf_display_type(const Arcane::Variable* var)
{
  arcane_ttf_header();
  _displayVariable(var);
  return TV_ttf_format_ok;
}

ATTR_USED int
TV_ttf_display_type(const Arcane::IVariable* var)
{
  arcane_ttf_header();
  _displayVariable(var);
  return TV_ttf_format_ok;
}

#define TV_DISPLAY_VARIABLE(var_type,value_type)\
ATTR_USED \
int \
TV_ttf_display_type(const Arcane:: var_type* obj)\
{\
  char strtype[1024];\
  arcane_ttf_header();\
  Arcane::IVariable* var = obj->variable();\
  TV_ttf_add_row("variable","Arcane::IVariable",var);\
  Arcane::ConstArrayView<value_type> view = obj->asArray();\
  snprintf(strtype,sizeof(strtype),"%s[%d]",#value_type,view.size());\
  TV_ttf_add_row("data",strtype,view.unguardedBasePointer());\
  return TV_ttf_format_ok;\
}

TV_DISPLAY_VARIABLE(VariableItemReal,Arcane::Real)
TV_DISPLAY_VARIABLE(VariableItemReal3,Arcane::Real3)
TV_DISPLAY_VARIABLE(VariableItemReal3x3,Arcane::Real3x3)
TV_DISPLAY_VARIABLE(VariableItemReal2,Arcane::Real2)
TV_DISPLAY_VARIABLE(VariableItemReal2x2,Arcane::Real2x2)
TV_DISPLAY_VARIABLE(VariableItemInt16,Arcane::Int16)
TV_DISPLAY_VARIABLE(VariableItemInt32,Arcane::Int32)
TV_DISPLAY_VARIABLE(VariableItemInt64,Arcane::Int64)
TV_DISPLAY_VARIABLE(VariableItemByte,Arcane::Byte)

#define TV_DISPLAY_VARIABLE_SCALAR(var_type,value_type)\
ATTR_USED \
int \
TV_ttf_display_type(const Arcane:: var_type* obj)\
{\
  char strtype[1024];\
  arcane_ttf_header();\
  Arcane::IVariable* var = obj->variable();\
  TV_ttf_add_row("variable","Arcane::IVariable",var);\
  Arcane::ConstArrayView<value_type> view = obj->asArray();\
  snprintf(strtype,sizeof(strtype),"%s",#value_type);\
  TV_ttf_add_row("data",strtype,view.data());\
  return TV_ttf_format_ok;\
}

TV_DISPLAY_VARIABLE_SCALAR(VariableScalarReal,Arcane::Real)
TV_DISPLAY_VARIABLE_SCALAR(VariableScalarReal3,Arcane::Real3)
TV_DISPLAY_VARIABLE_SCALAR(VariableScalarReal3x3,Arcane::Real3x3)
TV_DISPLAY_VARIABLE_SCALAR(VariableScalarReal2,Arcane::Real2)
TV_DISPLAY_VARIABLE_SCALAR(VariableScalarReal2x2,Arcane::Real2x2)
TV_DISPLAY_VARIABLE_SCALAR(VariableScalarInt16,Arcane::Int16)
TV_DISPLAY_VARIABLE_SCALAR(VariableScalarInt32,Arcane::Int32)
TV_DISPLAY_VARIABLE_SCALAR(VariableScalarInt64,Arcane::Int64)
TV_DISPLAY_VARIABLE_SCALAR(VariableScalarByte,Arcane::Byte)

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
