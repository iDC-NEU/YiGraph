import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from aag.models.graph_workflow_dag import GraphWorkflowDAG  # noqa: E402


class DummyReasoner:
    def __init__(self, revised_plan):
        self.revised_plan = revised_plan
        self.calls = []

    def revise_subquery_plan(self, current_plan, user_request, system_prompt=None):
        self.calls.append(
            {
                "current_plan": current_plan,
                "user_request": user_request,
                "system_prompt": system_prompt,
            }
        )
        return self.revised_plan


class TestDagRevision(unittest.TestCase):
    def setUp(self):
        self.dag = GraphWorkflowDAG()
        self.original_plan = {
            "subqueries": [
                {"id": "q1", "query": "Find Anna's community", "depends_on": []},
                {
                    "id": "q2",
                    "query": "Trace suspicious paths that involve Anna",
                    "depends_on": ["q1"],
                },
            ]
        }
        self.dag.build_from_subquery_plan(self.original_plan)

    def test_revise_subquery_plan_updates_plan_and_structure(self):
        revised_plan = {
            "subqueries": [
                {"id": "q1", "query": "Find Anna's community", "depends_on": []},
                {
                    "id": "q_inspect",
                    "query": "Identify fraud clusters within Anna's community",
                    "depends_on": ["q1"],
                },
                {
                    "id": "q2",
                    "query": "Trace suspicious paths that involve Anna",
                    "depends_on": ["q_inspect"],
                },
            ]
        }
        reasoner = DummyReasoner(revised_plan)
        user_request = "  Insert a check step before tracing suspicious paths.  "

        new_plan = self.dag.revise_subquery_plan(reasoner, user_request)

        self.assertEqual(new_plan, revised_plan)
        self.assertEqual(self.dag.get_subquery_plan(), revised_plan)
        self.assertEqual(len(reasoner.calls), 1)
        call = reasoner.calls[0]
        self.assertEqual(call["current_plan"], self.original_plan)
        self.assertEqual(
            call["user_request"], user_request.strip(), "Request should be normalized"
        )

        # DAG should be rebuilt to match the revised plan
        self.assertEqual(len(self.dag.steps), 3)
        self.assertEqual(len(self.dag.topological_order()), 3)
        query_mapping = self.dag.get_query_id_mapping()
        inspect_step_id = query_mapping["q_inspect"]
        self.assertListEqual(
            sorted(self.dag.parents_of(inspect_step_id)), [query_mapping["q1"]]
        )
        q2_step_id = query_mapping["q2"]
        self.assertListEqual(sorted(self.dag.parents_of(q2_step_id)), [inspect_step_id])


if __name__ == "__main__":
    unittest.main()
