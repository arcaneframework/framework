<?xml version="1.0"?>
<case codename="csv" xml:lang="en" codeversion="1.0">
  <arcane>
    <title>Examples CSV</title>
    <timeloop>examples2_6</timeloop>
  </arcane>

  <meshes>
    <mesh>
      <generator name="Cartesian2D" >
        <face-numbering-version>1</face-numbering-version>

        <nb-part-x>1</nb-part-x> 
        <nb-part-y>1</nb-part-y>

        <origin>0.0 0.0</origin>

        <x>
          <length>1.0</length>
          <n>1</n>
        </x>

        <y>
          <length>1.0</length>
          <n>1</n>
        </y>

      </generator>
    </mesh>
  </meshes>

  <simple-table-output-example2>
    <!-- Attention, les noms sont utilisés par le module SimpleTableOutputExample2Module,
    pas directement par le service CSV. -->
    <!-- Le nom du répertoire à créer/utiliser. -->
    <tableDir>example2</tableDir>
    <!-- Le nom du fichier à créer/écraser. -->
    <tableName>Results_Example2</tableName>

    <!-- Au final, on aura un fichier ayant comme chemin : ./example2/Results_Example2.csv -->
  </simple-table-output-example2>

  <simple-table-output-example3>
    <st-output name="SimpleCsvOutput">
      <!-- Les noms sont utilisés directement par le service CSV. -->
      <!-- Le nom du répertoire à créer/utiliser. -->
      <tableDir>example3</tableDir>
      <!-- Le nom du fichier à créer/écraser. -->
      <tableName>Results_Example3</tableName>

      <!-- Au final, on aura un fichier ayant comme chemin : ./example3/Results_Example3.csv -->
    </st-output>
  </simple-table-output-example3>

  <simple-table-output-example4>
    <st-output name="SimpleCsvOutput">
      <tableDir>example4</tableDir>
      <tableName>Results_Example4</tableName>
    </st-output>
  </simple-table-output-example4>

  <simple-table-output-example5>
    <st-output name="SimpleCsvOutput">
      <tableDir>example5</tableDir>
      <tableName>Results_Example5</tableName>
    </st-output>
  </simple-table-output-example5>

  <simple-table-output-example6>
    <st-output name="SimpleCsvOutput">
      <tableDir>example6</tableDir>
      <tableName>Results_Example6</tableName>
    </st-output>
  </simple-table-output-example6>

</case>
