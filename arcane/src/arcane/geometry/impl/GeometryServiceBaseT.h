// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include "arcane/geometry/impl/GeometryTemplatesT.h"
#include "arcane/VariableBuildInfo.h"

ARCANE_BEGIN_NAMESPACE
NUMERICS_BEGIN_NAMESPACE

template<typename GeometryT>
void 
GeometryServiceBase::
updateGroup(ItemGroup group, GeometryT & geometry) 
{
  PropertyMap::iterator igroup = m_group_property_map.find(group.internal());
  if (igroup == m_group_property_map.end())
    throw FatalErrorException(A_FUNCINFO,"Undefined ItemGroup property");

  ItemGroupGeometryProperty & properties = igroup->second;
  if ((properties.defined & ~properties.computed & ~properties.delegated) == 0)
    {
      traceMng()->debug() << "Group " << group.name() << " properties already done";
      return;
    }

  for(typename ItemGroupGeometryProperty::StorageInfos::iterator i = properties.storages.begin(); i != properties.storages.end(); ++i) 
    {
      IGeometryProperty::eProperty property = i->first;
      ItemGroupGeometryProperty::StorageInfo & storage = i->second;

      if ((properties.computed & property) || (properties.delegated & property))
        {
          traceMng()->debug() << "Property " << IGeometryProperty::name(property) << " is delayed for group " << group.name();
          continue; // skip that property
        }
      else        
        {
          traceMng()->debug() << "Property " << IGeometryProperty::name(property) << " will be computed for group " << group.name();
        }

      if (IGeometryProperty::isScalar(property)) 
        {
          std::shared_ptr<RealVariable> & ivar = storage.realVar;
          if (!ivar && (storage.storageType & IGeometryProperty::PVariable))
            {
              String varName = IGeometryProperty::name(property)+String("Of")+group.name()+m_suffix;
              traceMng()->debug() << "Building Variable " << varName;
              ivar.reset(new RealVariable(VariableBuildInfo(group.mesh(),
																		varName,
																		group.itemFamily()->name(),
                                                            IVariable::PPrivate|IVariable::PTemporary),
                                          group.itemKind()));
            }            

        }
      else
        {
          ARCANE_ASSERT((IGeometryProperty::isVectorial(property)),("Vectorial property expected"));
          std::shared_ptr<Real3Variable> & ivar = storage.real3Var;
          if (!ivar && (storage.storageType & IGeometryProperty::PVariable))
            {
              String varName = IGeometryProperty::name(property)+String("Of")+group.name()+m_suffix;
              traceMng()->debug() << "Building Variable " << varName;
              ivar.reset(new Real3Variable(VariableBuildInfo(group.mesh(),
																		varName,
																		group.itemFamily()->name(),
                                                             IVariable::PPrivate|IVariable::PTemporary),
                                           group.itemKind()));
            }            

        }
    }

  if (properties.defined & ~properties.computed & ~properties.delegated) {
    GenericGSInternalUpdater<GeometryT> updater(geometry, traceMng());
    updater.setGroupProperty(&properties);
    group.applyOperation(&updater);
  }

  /*** L'affectation générale est désormais faite dans GeometryTemplatesT ***/
  for(typename ItemGroupGeometryProperty::StorageInfos::iterator i = properties.storages.begin(); i != properties.storages.end(); ++i) 
    {
      IGeometryProperty::eProperty property = i->first;
      if ((properties.computed & property) || (properties.delegated & property))
        continue;
      properties.computed |= property;
    }
}

NUMERICS_END_NAMESPACE
ARCANE_END_NAMESPACE
