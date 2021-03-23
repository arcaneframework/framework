#ifndef ITEMCOMPARATOR_H
#define ITEMCOMPARATOR_H

struct ItemComparator {
  bool operator()(const Item & a, const Item & b) const {
    ARCANE_ASSERT((a.kind() == b.kind()),("Incompatible item kinds: %d vs %d",a.kind(),b.kind()));
    return a.localId() < b.localId();
  }
};

#endif
