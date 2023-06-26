#include <ALIEN/Alien-RefSemanticMVHandlers.h>
#include <ALIEN/IndexManager/Functional/DefaultIndexManager.h>

namespace Environment {
extern Alien::ITraceMng* traceMng();
}

void
buildMatrix(Alien::Matrix& A, std::string const& filename, std::string const& format)
{
  auto* tm = Environment::traceMng();

  // Distributions calculée
  const auto& dist = A.distribution();
  int offset = dist.rowOffset();
  int lsize = dist.localRowSize();

  auto* pm = dist.parallelMng();

  if (lsize % 2)
    tm->fatal() << "local size should be pair";

  int size = lsize / 2;

  tm->info() << "build uids for left part";

  Alien::Int64UniqueArray uids_left(size, -1);
  for (auto i = 0; i < size; ++i)
    uids_left[i] = offset + i;

  tm->info() << "build uids for right part";

  Alien::Int64UniqueArray uids_right(size, -1);
  for (auto i = 0; i < size; ++i)
    uids_right[i] = offset + size + i;

  tm->info() << "build basic index manager";

  Alien::DefaultIndexManager im(pm, { uids_left, uids_right });

  tm->info() << "build matrix with direct matrix builder";
  {
    Alien::DirectMatrixBuilder builder(A, Alien::DirectMatrixOptions::eResetValues);
    builder.reserve(3); // Réservation de 3 coefficients par ligne
    builder.allocate(); // Allocation de l'espace mémoire réservé

    auto left = im[0];
    auto right = im[1];

    auto rank = pm->commRank();

    for (int i = 0; i < size; ++i) {
      // matrices locales
      builder(left[i], left[i]) = 2.;
      builder(right[i], right[i]) = 2.;
      if (i - 1 >= 0) {
        builder(left[i], left[i] - 1) = -1.;
        builder(right[i], right[i] - 1) = -1.;
      }
      if (i + 1 < size) {
        builder(left[i], left[i] + 1) = -1.;
        builder(right[i], right[i] + 1) = -1.;
      }
      // connection entre matrices
      builder(left[size - 1], right[0]) = -1.;
      builder(right[0], left[size - 1]) = -1.;
      // connection parallèle
      if (rank > 0) {
        builder(left[0], left[0] - 1) = -1.;
      }
      if (rank < pm->commSize() - 1) {
        builder(right[size - 1], left[size - 1] + 1) = -1.;
      }
    }
  }
}
