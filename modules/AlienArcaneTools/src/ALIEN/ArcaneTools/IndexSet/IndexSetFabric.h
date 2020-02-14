#ifndef ALIEN_ARCANETOOLS_INDEXSET_INDEXSETFABRIC_H
#define ALIEN_ARCANETOOLS_INDEXSET_INDEXSETFABRIC_H

#include <ALIEN/Data/Space.h>
#include <ALIEN/ArcaneTools/IIndexManager.h>

#include <arccore/trace/ITraceMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  inline void createIndexSet(Space& space, IIndexManager* index_mng, Arccore::String tag,
      Arccore::ITraceMng* trace_mng)
  {
    Arccore::UniqueArray<Arccore::Integer> current_field_indices;

    for(auto i = index_mng->enumerateEntry(); i.hasNext(); ++i)
  {
    if((*i).tagValue("block-tag") == tag)
    {
      trace_mng->info() << "block-tag '" << tag << "' will tag entry '" << i->getName() << "'";

      auto indices = i->getOwnIndexes();

      current_field_indices.addRange(indices);
    }
  }

  trace_mng->info() << "block-tag '" << tag << "' has " << current_field_indices.size() << " indices";

  if(current_field_indices.size() == 0) return;

  space.setField(tag, current_field_indices);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_ARCANETOOLS_INDEXSET_INDEXSETFABRIC_H */
