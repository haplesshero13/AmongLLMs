"""
Log parsing utilities to extract structured game data from logs and update summaries.
"""

import json
import sys
from typing import Dict, List, Any, Optional


class GameLogParser:
    """Parse and enhance Among Us game logs with structured voting and kill data."""

    def __init__(self, game_log: Dict[str, Any]):
        """Initialize with a game log JSON object."""
        self.game_log = game_log
        self.agent_logs = game_log.get("agent_logs", [])
        self.summary = game_log.get("summary", {})

    def extract_voting_history(self) -> List[Dict[str, Any]]:
        """Extract voting history from agent logs."""
        voting_history = []

        # Group votes by timestep
        votes_by_step = {}

        for log_entry in self.agent_logs:
            # Extract action from response dict - try multiple paths
            response = log_entry.get("interaction", {}).get("response", {})
            action = ""
            if isinstance(response, dict):
                # Path 1: response.Action (top-level field)
                action = response.get("Action", "")
                # Path 2: response["Thinking Process"].action (nested)
                if not action and "Thinking Process" in response:
                    thinking_process = response.get("Thinking Process", {})
                    action = (
                        thinking_process.get("action", "")
                        if isinstance(thinking_process, dict)
                        else ""
                    )

            # Check if this is a vote action
            if action and "VOTE" in action:
                # Parse: "VOTE Player 5: blue"
                parts = action.split("VOTE", 1)  # Only split once
                if len(parts) >= 2:
                    target = parts[1].strip()
                    # Extract just the player name (format: "Player X: color")
                    import re

                    match = re.match(r"(Player \d+: \w+)", target)
                    if match:
                        target = match.group(1)

                    voter = log_entry.get("player", {}).get("name", "")
                    timestep = log_entry.get("step")

                    if timestep not in votes_by_step:
                        votes_by_step[timestep] = []

                    votes_by_step[timestep].append(
                        {"voter": voter, "target": target, "timestep": timestep}
                    )

        # Convert each voting step into a voting round
        meeting_number = 0
        for timestep in sorted(votes_by_step.keys()):
            meeting_number += 1
            votes = votes_by_step[timestep]

            # Calculate vote tally
            vote_tally = {}
            for vote in votes:
                target = vote["target"]
                vote_tally[target] = vote_tally.get(target, 0) + 1

            # Determine eliminated player (most votes)
            eliminated = None
            if vote_tally:
                max_votes = max(vote_tally.values())
                players_with_max = [
                    player for player, count in vote_tally.items() if count == max_votes
                ]
                eliminated = players_with_max[0] if len(players_with_max) == 1 else None

            voting_round = {
                "meeting_number": meeting_number,
                "timestep": timestep,
                "votes": votes,
                "vote_tally": vote_tally,
                "eliminated": eliminated,
            }
            voting_history.append(voting_round)

        return voting_history

        return voting_history

    def extract_kill_history(self) -> List[Dict[str, Any]]:
        """Extract kill history from agent logs."""
        kill_history = []

        for log_entry in self.agent_logs:
            # Extract action from response dict - try multiple paths
            response = log_entry.get("interaction", {}).get("response", {})
            action = ""
            if isinstance(response, dict):
                # Path 1: response.Action (top-level field)
                action = response.get("Action", "")
                # Path 2: response["Thinking Process"].action (nested)
                if not action and "Thinking Process" in response:
                    thinking_process = response.get("Thinking Process", {})
                    action = (
                        thinking_process.get("action", "")
                        if isinstance(thinking_process, dict)
                        else ""
                    )

            # Check if this is a kill action
            if action and "KILL" in action:
                # Parse: "KILL Player 4: red"
                parts = action.split("KILL", 1)  # Only split once
                if len(parts) >= 2:
                    victim = parts[1].strip()
                    # Extract just the player name (format: "Player X: color")
                    import re

                    match = re.match(r"(Player \d+: \w+)", victim)
                    if match:
                        victim = match.group(1)

                    killer = log_entry.get("player", {}).get("name", "")
                    timestep = log_entry.get("step")
                    location = log_entry.get("player", {}).get("location", "unknown")

                    kill_record = {
                        "killer": killer,
                        "victim": victim,
                        "timestep": timestep,
                        "location": location,
                    }
                    kill_history.append(kill_record)

        return kill_history

    def extract_game_outcome(self) -> Dict[str, Any]:
        """Extract enhanced game outcome information."""
        winner = self.summary.get("winner")
        winner_reason = self.summary.get("winner_reason", "")

        # Get all players from summary
        all_players = {}
        for player_key, player_info in self.summary.items():
            if isinstance(player_info, dict) and "identity" in player_info:
                player_name = player_info["name"]
                all_players[player_name] = player_info["identity"]

        # Collect eliminated players from voting and kill history
        eliminated_set = set()

        # Get voting history
        voting_history = self.extract_voting_history()
        for voting_round in voting_history:
            eliminated = voting_round.get("eliminated")
            if eliminated:
                eliminated_set.add(eliminated)

        # Get kill history
        kill_history = self.extract_kill_history()
        for kill_record in kill_history:
            victim = kill_record.get("victim")
            if victim:
                eliminated_set.add(victim)

        # Determine surviving players
        eliminated_players = sorted(list(eliminated_set))
        surviving_players = sorted(
            [p for p in all_players.keys() if p not in eliminated_set]
        )

        # Count final impostor/crewmate numbers among survivors
        final_impostor_count = sum(
            1 for p in surviving_players if all_players.get(p, "").lower() == "impostor"
        )
        final_crewmate_count = len(surviving_players) - final_impostor_count

        # Validate game outcome consistency
        self._validate_game_outcome(
            winner_reason,
            final_impostor_count,
            final_crewmate_count,
            surviving_players,
            eliminated_players,
            all_players,
        )

        return {
            "winner": winner,
            "reason": winner_reason,
            "surviving_players": surviving_players,
            "eliminated_players": eliminated_players,
            "final_impostor_count": final_impostor_count,
            "final_crewmate_count": final_crewmate_count,
        }

    def _validate_game_outcome(
        self,
        winner_reason: str,
        final_impostor_count: int,
        final_crewmate_count: int,
        surviving_players: List[str],
        eliminated_players: List[str],
        all_players: Dict[str, str],
    ):
        """Validate that extracted game outcome matches the stated winner reason."""
        warnings = []

        # Check if outcome matches the winner reason
        if (
            "impostor" in winner_reason.lower()
            and "eliminated" in winner_reason.lower()
        ):
            # Should have 0 impostors surviving
            if final_impostor_count > 0:
                impostor_survivors = [
                    p
                    for p in surviving_players
                    if all_players.get(p, "").lower() == "impostor"
                ]
                warnings.append(
                    f"Game claims 'Impostors eliminated' but {final_impostor_count} impostor(s) survived: {impostor_survivors}"
                )

        elif "impostor" in winner_reason.lower() and "win" in winner_reason.lower():
            # Impostors won - should have impostors alive
            if final_impostor_count == 0:
                warnings.append("Game claims 'Impostors win' but no impostors survived")

        elif "task" in winner_reason.lower() and "complete" in winner_reason.lower():
            # Tasks completed - would need task tracking to validate this
            warnings.append(
                "Game claims 'Tasks completed' but task completion tracking not implemented in parser"
            )

        # Check for missing players
        total_players = len(surviving_players) + len(eliminated_players)
        expected_players = len(all_players)
        if total_players != expected_players:
            warnings.append(
                f"Player count mismatch: {total_players} accounted for vs {expected_players} expected"
            )

        # Output warnings to stderr
        if warnings:
            print("\n⚠️  WARNING: Inconsistent game outcome detected:", file=sys.stderr)
            for warning in warnings:
                print(f"   - {warning}", file=sys.stderr)
            print(
                "   This may indicate incomplete/malformed LLM outputs in the original game logs.\n",
                file=sys.stderr,
            )

    def generate_enhanced_summary(self) -> Dict[str, Any]:
        """Generate enhanced summary with voting, kill, and outcome data."""
        enhanced_summary = dict(self.summary)

        enhanced_summary["voting_history"] = self.extract_voting_history()
        enhanced_summary["kill_history"] = self.extract_kill_history()
        enhanced_summary["game_outcome"] = self.extract_game_outcome()

        return enhanced_summary

    def update_game_log(self) -> Dict[str, Any]:
        """Return updated game log with enhanced summary."""
        updated_log = dict(self.game_log)
        updated_log["summary"] = self.generate_enhanced_summary()
        return updated_log


def update_existing_game_log(
    file_path: str, output_path: Optional[str] = None
) -> Dict[str, Any]:
    """Update an existing game log file with enhanced summary data."""
    with open(file_path, "r") as f:
        game_log = json.load(f)

    parser = GameLogParser(game_log)
    updated_log = parser.update_game_log()

    output_path = output_path or file_path
    with open(output_path, "w") as f:
        json.dump(updated_log, f, indent=2)

    return updated_log


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python log_parser.py <game_log_file.json> [output_file.json]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file

    try:
        updated_log = update_existing_game_log(input_file, output_file)
        print(f"Successfully updated game log: {output_file}")
        print(f"Added {len(updated_log['summary']['voting_history'])} voting rounds")
        print(f"Added {len(updated_log['summary']['kill_history'])} kill records")
    except Exception as e:
        print(f"Error updating game log: {e}")
        sys.exit(1)
