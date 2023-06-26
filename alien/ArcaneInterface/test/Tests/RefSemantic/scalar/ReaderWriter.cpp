#include <alien/ref/AlienRefSemantic.h>
#include <Tests/Options.h>

int
main(int argc, char** argv)
{
  // Pour initialiser MPI, les traces et
  // le gestionnaire de parallélisme
  return Environment::execute(argc, argv, [&] {

    auto* pm = Environment::parallelMng();
    auto* tm = Environment::traceMng();

    tm->info() << "Example Alien :";
    tm->info() << "Use of Alien::IVector readers / writers (RefSemanticMVHandlers API)";
    tm->info() << " ";
    tm->info() << "Start example";
    tm->info() << " ";

    // Le gestionnaire de trace est donné à Alien
    Alien::setTraceMng(tm);

    // On définit le niveau de verbosité
    Alien::setVerbosityLevel(Alien::Verbosity::Debug);

    // Taille globale
    int size = 100;

    // Objects algébriques

    tm->info() << "define vectors";

    Alien::Vector b(size, pm);

    // Distributions calculée
    const auto& dist = b.distribution();
    int offset = dist.offset();
    int lsize = dist.localSize();

    // Mode de remplissage par global ids
    tm->info() << "fill vectors with Alien::VectorWriter";
    {
      // Builder du vecteur
      Alien::VectorWriter writer(b);

      // On remplit le vecteur
      for (int i = 0; i < lsize; ++i) {
        writer[i + offset] = i + offset + 1;
      }
    }
    // Mode de lecture par global ids
    tm->info() << "read vectors with Alien::VectorReader";
    { // NB: Scope non utile
      // Lecteur du vecteur
      Alien::VectorReader reader(b);

      for (int i = 0; i < lsize; ++i) {
        if (reader[i + offset] != i + offset + 1)
          throw Alien::FatalErrorException("VectorWriter/VectorReader error");
      }
    }
    // Mode de lecture par local ids
    tm->info() << "read vectors with Alien::LocalVectorReader";
    {
      // Lecteur du vecteur
      Alien::LocalVectorReader reader(b);

      for (int i = 0; i < lsize; ++i) {
        if (reader[i] != i + offset + 1)
          throw Alien::FatalErrorException("VectorWriter/LocalVectorReader error");
      }
    }
    // Mode de remplissage par local ids
    tm->info() << "fill vectors with Alien::LocalVectorWriter";
    {
      // Builder du vecteur
      Alien::LocalVectorWriter writer(b);

      // On remplit le vecteur
      for (int i = 0; i < lsize; ++i) {
        writer[i] = i + offset + 10;
      }
    }
    // Mode de lecture par global ids
    tm->info() << "read vectors with Alien::VectorReader";
    {
      // Lecteur du vecteur
      Alien::VectorReader reader(b);

      for (int i = 0; i < lsize; ++i) {
        if (reader[i + offset] != i + offset + 10)
          throw Alien::FatalErrorException("LocalVectorWriter/VectorReader error");
      }
    }
    // Mode de lecture par local ids
    tm->info() << "read vectors with Alien::LocalVectorReader";
    {
      // Lecteur du vecteur
      Alien::LocalVectorReader reader(b);

      for (int i = 0; i < lsize; ++i) {
        if (reader[i] != i + offset + 10)
          throw Alien::FatalErrorException("LocalVectorWriter/LocalVectorReader error");
      }
    }

    tm->info() << " ";
    tm->info() << "Example finished!";

    return 0;
  });
}
