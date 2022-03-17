import unittest
from slam.BayesTree import BayesTree, BayesTreeNode
from slam.Variables import Variable


class BeysTreeStructureTestCase1(unittest.TestCase):
    def setUp(self) -> None:
        """
        Set up the following 2D Gaussian example:
            x2 -- (1, 1) -- x0 -- (0, -1) -- x3 -- (2, 1) -- x1
                -- (-2, -1) -- x4
        """
        self.l1 = Variable("L1", 2)
        self.l2 = Variable("L2", 2)
        self.x0 = Variable("X0", 4)
        self.x1 = Variable("X1", 4)
        self.x2 = Variable("X2", 4)
        self.x3 = Variable("X3", 4)
        self.clique3 = BayesTreeNode(frontal={self.x1, self.x2})
        self.clique1 = BayesTreeNode(frontal={self.x0, self.l1},
                                     separator={self.x1})
        self.clique2 = BayesTreeNode(frontal={self.l2}, separator={self.x2})
        self.clique3.append_child(self.clique1)
        self.clique3.append_child(self.clique2)
        self.tree1 = BayesTree(self.clique3)

    def test_constructor_from_clique(self) -> None:
        self.assertEqual(self.tree1.root, self.clique3)
        self.assertCountEqual(self.tree1.leaves, {self.clique1,
                                                  self.clique2})
        self.assertCountEqual(self.clique1.children, {})
        self.assertCountEqual(self.clique2.children, {})
        self.assertCountEqual(self.tree1.leaves, {self.clique1,
                                                  self.clique2})
        self.assertEqual(self.clique1.parent, self.clique3)
        self.assertEqual(self.clique2.parent, self.clique3)
        self.assertEqual(self.clique3.parent, None)
        self.assertCountEqual(self.tree1.clique_nodes, {self.clique1,
                                                        self.clique2,
                                                        self.clique3})

    def test_copy_node(self) -> None:
        node1 = Variable("1", 1)
        node2 = Variable("2", 2)
        node3 = Variable("3", 1)
        old_clique = BayesTreeNode(frontal={node1, node2}, separator={node3})
        new_clique = old_clique.copy_without_parents_children()
        node4 = Variable("4", 2)
        new_clique.add_frontal(node4)
        self.assertCountEqual(old_clique.frontal, {node1, node2})
        self.assertCountEqual(new_clique.frontal, {node1, node2, node4})

    def test_copy_bayes_tree(self) -> None:
        copy = self.tree1.__copy__()
        root = copy.root
        self.assertCountEqual(root.frontal, self.clique3.frontal)
        self.assertCountEqual(root.separator, self.clique3.separator)
        self.assertEqual(root.parent, None)
        for child in root.children:
            self.assertEqual(child.parent, root)
            self.assertCountEqual(child.children, {})
            self.assertTrue((child.frontal == {self.x0, self.l1} and
                             child.separator == {self.x1}) or
                            (child.frontal == {self.l2} and
                             child.separator == {self.x2}))
        self.assertEqual(self.clique3, root)
        x3 = Variable("X3", 4)
        root.add_frontal(x3)
        self.assertCountEqual(self.clique3.frontal, {self.x1, self.x2})
        self.assertCountEqual(root.frontal,
                              set.union(self.clique3.frontal, {x3}))

    def test_affected_part_1(self) -> None:
        # A new factor is connected to X2
        affected_vars, sub_trees = \
            self.tree1.get_affected_vars_and_partial_bayes_trees({self.x2})
        self.assertCountEqual(set(affected_vars),
                              set(self.clique3.frontal))
        for sub_tree in sub_trees:
            if sub_tree.clique_nodes == {self.clique1}:
                self.assertEqual(sub_tree.root, self.clique1)
                self.assertEqual(sub_tree.leaves, {self.clique1})
            elif sub_tree.clique_nodes == {self.clique2}:
                self.assertEqual(sub_tree.root, self.clique2)
                self.assertEqual(sub_tree.leaves, {self.clique2})
            else:
                self.assertTrue(False)
        new_clique3 = BayesTreeNode(frontal={self.x1}, separator={self.x2})
        new_clique4 = BayesTreeNode(frontal={self.x2, self.x3})
        new_tree = BayesTree(new_clique4)
        new_tree.append_clique(clique=new_clique3, parent_clique=new_clique4)
        new_tree.append_child_bayes_trees(sub_trees)
        self.assertEqual(new_clique4, new_tree.root)
        self.assertCountEqual(new_tree.clique_nodes, {self.clique1,
                                                      self.clique2,
                                                      new_clique3, new_clique4})
        self.assertCountEqual(new_tree.leaves, {self.clique1, self.clique2})
        self.assertCountEqual(new_clique3.children, {self.clique1})
        self.assertCountEqual(new_clique4.children, {new_clique3, self.clique2})

    def test_affected_part_2(self) -> None:
        # A new factor is connected to X1
        affected_vars, sub_trees = \
            self.tree1.get_affected_vars_and_partial_bayes_trees({self.x1})
        self.assertCountEqual(set(affected_vars),
                              set(self.clique3.frontal))
        for sub_tree in sub_trees:
            if sub_tree.clique_nodes == {self.clique1}:
                self.assertEqual(sub_tree.root, self.clique1)
                self.assertEqual(sub_tree.leaves, {self.clique1})
            elif sub_tree.clique_nodes == {self.clique2}:
                self.assertEqual(sub_tree.root, self.clique2)
                self.assertEqual(sub_tree.leaves, {self.clique2})
            else:
                self.assertTrue(False)
        new_clique3 = BayesTreeNode(frontal={self.x1, self.x2, self.x3})
        new_tree = BayesTree(root_clique=new_clique3)
        new_tree.append_child_bayes_trees(sub_trees)
        self.assertEqual(new_clique3, new_tree.root)
        self.assertCountEqual(new_clique3.children, {self.clique1,
                                                     self.clique2})
        self.assertCountEqual(new_tree.clique_nodes, {self.clique1,
                                                      self.clique2,
                                                      new_clique3})
        self.assertCountEqual(new_tree.leaves, {self.clique1, self.clique2})

    def test_affected_part_3(self) -> None:
        # A new factor is connected to X0
        affected_vars, sub_trees = \
            self.tree1.get_affected_vars_and_partial_bayes_trees({self.x0})
        self.assertCountEqual(set(affected_vars),
                              set.union(self.clique1.frontal,
                                        self.clique3.frontal))
        for sub_tree in sub_trees:
            self.assertTrue(sub_tree.clique_nodes == {self.clique2})

    def test_affected_part_4(self) -> None:
        # A new factor is connected to L1
        affected_vars, sub_trees = \
            self.tree1.get_affected_vars_and_partial_bayes_trees({self.l1})
        self.assertCountEqual(set(affected_vars),
                              set.union(self.clique1.frontal,
                                        self.clique3.frontal))
        for sub_tree in sub_trees:
            self.assertTrue(sub_tree.clique_nodes == {self.clique2})

    def test_affected_part_5(self) -> None:
        # A new factor is connected to L2
        affected_vars, sub_trees = \
            self.tree1.get_affected_vars_and_partial_bayes_trees({self.l2})
        self.assertCountEqual(set(affected_vars),
                              set.union(self.clique2.frontal,
                                        self.clique3.frontal))
        for sub_tree in sub_trees:
            self.assertTrue(sub_tree.clique_nodes == {self.clique1})


class BeysTreeStructureTestCase2(unittest.TestCase):
    def test(self) -> None:
        x0 = Variable("X0", 4)
        x1 = Variable("X1", 4)
        x2 = Variable("X2", 4)
        x3 = Variable("X3", 4)
        x4 = Variable("X4", 4)
        x5 = Variable("X5", 4)
        x6 = Variable("X6", 4)
        x7 = Variable("X7", 4)
        x8 = Variable("X8", 4)
        x9 = Variable("X9", 4)
        clique8 = BayesTreeNode(frontal={x8}, separator={x1})
        clique12 = BayesTreeNode(frontal={x1, x2}, separator={x3})
        clique0 = BayesTreeNode(frontal={x0}, separator={x4})
        clique34 = BayesTreeNode(frontal={x3, x4}, separator={x5, x6})
        clique567 = BayesTreeNode(frontal={x5, x6, x7})
        clique567.append_child(clique34)
        clique34.append_child(clique12)
        clique34.append_child(clique0)
        clique12.append_child(clique8)
        old_tree = BayesTree(root_clique=clique567)
        affected_nodes, sub_trees = \
            old_tree.get_affected_vars_and_partial_bayes_trees({x2})
        self.assertCountEqual(set(affected_nodes),
                              {x1, x2, x3, x4, x5, x6, x7})
        clique14 = BayesTreeNode(frontal={x1, x4})
        new_tree = BayesTree(root_clique=clique14)
        new_tree.append_child_bayes_trees(sub_trees)
        self.assertEqual(new_tree.root, clique14)
        self.assertCountEqual(set(new_tree.leaves), {clique0, clique8})
        self.assertCountEqual(set(new_tree.root.children), {clique0, clique8})
        self.assertCountEqual(set(new_tree.clique_nodes),
                              {clique14, clique0, clique8})


if __name__ == '__main__':
    unittest.main()
