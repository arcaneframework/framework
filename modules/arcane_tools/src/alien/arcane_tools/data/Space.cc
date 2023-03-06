
#include "Space.h"
#include "alien/arcane_tools/IIndexManager.h"

#include <map>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Space::
Space(IIndexManager* index_mng, Arccore::String name)
: m_index_mng(index_mng)
{
  m_internal = std::make_shared<Alien::Space>(m_index_mng->globalSize(),name);
  this->_init();
}

/*---------------------------------------------------------------------------*/
  
Space::
Space(IIndexManager* index_mng, Integer block_size, Arccore::String name)
: m_index_mng(index_mng)
{
  m_internal = std::make_shared<Alien::Space>(m_index_mng->globalSize()/block_size), name;
  this->_init();
}

/*---------------------------------------------------------------------------*/

void
Space::
_init()
{
  if(not m_index_mng->isPrepared())
    m_index_mng->prepare();

  std::map<Arccore::String,Arccore::UniqueArray<Integer> > current_field_indices;
    
  for(auto i = m_index_mng->enumerateEntry(); i.hasNext(); ++i) {
    if(i->hasTag("block-tag")) {
      auto tag = i->tagValue("block-tag");
      auto indices = i->getOwnIndexes();
      Alien::addRange(current_field_indices[tag], indices);
    }
  }
    
  if(current_field_indices.size() == 0) return;
    
  for(auto i = current_field_indices.begin(); i != current_field_indices.end(); ++i) {
    if(i->second.size() > 0)
      m_internal->setField(i->first, i->second);
  }    
}
  
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

}


