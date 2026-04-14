import re
import pandas as pd
from data import load_all_games


def _extract_tasks(all_info_str):
    """Parse task names out of the 'YOUR ASSIGNED TASKS' section in All Info."""
    if not all_info_str:
        return []
    match = re.search(r'YOUR ASSIGNED TASKS:\n(.*?)(?:\n\nYOUR AVAILABLE ACTIONS|\Z)', all_info_str, re.DOTALL)
    if not match:
        return []
    tasks = []
    for line in match.group(1).splitlines():
        m = re.match(r'^\d+\.\s+\w+:\s+(.+?)(?:\s+\[completed\])?\s*$', line.strip(), re.IGNORECASE)
        if m:
            tasks.append(m.group(1).strip())
    return tasks

def infer_phase(action):
    if pd.isna(action):
        return 'Unknown'
    if any(k in action.upper() for k in ['SPEAK', 'VOTE']):
        return 'Meeting'
    return 'Task'


def create_meeting_transcripts(df):
    meeting_mask = df['interaction.response.Action'].str.contains('SPEAK|VOTE', case=False, na=False)
    meeting_data = df[meeting_mask].copy()
    meeting_transcripts = {}

    for step, group in meeting_data.groupby('step'):
        transcript = []
        group_sorted = group.sort_values('timestamp')

        for idx, row in group_sorted.iterrows():
            player_name = row['player.name']
            player_model = row['player.model']
            speech = row['interaction.response.Action']

            if pd.notna(speech):
                transcript.append(f"{player_name.ljust(20)} ({player_model.ljust(35)}): {speech}")

        meeting_transcripts[f'Meeting at Step {step}'] = '\n'.join(transcript)

    return meeting_transcripts


def create_player_narrative(player_df, player_name):
    narrative = [f"PLAYER NARRATIVE: {player_name}"]
    player_df_sorted = player_df.sort_values(['step', 'timestamp']).reset_index(drop=True)

    for idx, row in player_df_sorted.iterrows():
        step = row['step']
        action = row['interaction.response.Action']
        phase = infer_phase(action)
        narrative.append(f"STEP {step} | Phase: {phase}")

        thinking = row['interaction.response.Thinking Process']
        if pd.notna(thinking) and thinking.strip():
            narrative.append("REASONING:")
            narrative.append(thinking.strip())

        if pd.notna(action) and action.strip():
            narrative.append("ACTION:")
            narrative.append(action.strip())

        narrative.append("-" * 40)

    return '\n'.join(narrative)


def parse_game_logs(entries: list[dict]) -> dict:
    """Parse raw log entries for one game into narratives and transcripts.

    Args:
        entries: List of log-entry dicts as returned by data.fetch_game_logs()
                 or a value from data.load_all_games() / data.load_new_games().

    Returns:
        {
            'player_narratives':   {'Player 1': str, 'Player 2': str, ...},
            'meeting_transcripts': {'Meeting at Step N': str, ...},
        }
    """
    agent_logs = pd.json_normalize(entries)
    agent_logs_copy = agent_logs.copy(deep=True)

    cols_to_drop = [
        'player.personality', 'interaction.system_prompt',
        'interaction.prompt.All Info', 'interaction.prompt.Available Actions',
        'interaction.prompt.Current Step', 'interaction.prompt.Current Player',
        'interaction.response.Condensed Memory', 'interaction.full_response',
    ]
    agent_logs_copy.drop(
        columns=[c for c in cols_to_drop if c in agent_logs_copy.columns],
        inplace=True,
    )

    unique_players = agent_logs_copy['player.name'].unique()
    players = {}
    for i, player in enumerate(unique_players, start=1):
        player_df = agent_logs_copy[agent_logs_copy['player.name'] == player].copy()
        drop_cols = ['player.name', 'player.identity', 'game_index', 'game_id']
        player_df.drop(
            columns=[c for c in drop_cols if c in player_df.columns],
            inplace=True,
        )
        players[f'Player {i}'] = player_df

    meeting_transcripts = create_meeting_transcripts(agent_logs_copy)

    player_narratives = {
        key: create_player_narrative(df, key)
        for key, df in players.items()
    }

    return {
        'player_narratives': player_narratives,
        'meeting_transcripts': meeting_transcripts,
        'agent_logs_df': agent_logs_copy,
        'players': players,
    }

def create_game_log(data, agent_logs_df, players_dict, meeting_transcripts):
    # data is the raw entries list; agent_logs_df has All Info dropped,
    # so we read tasks from the original entries.
    player_tasks = {}
    for entry in data:
        name = entry.get('player', {}).get('name')
        if name and name not in player_tasks:
            all_info = entry.get('interaction', {}).get('prompt', {}).get('All Info', '')
            player_tasks[name] = _extract_tasks(all_info)

    unique_players = agent_logs_df['player.name'].unique()
    players_list = []

    for i, player_name in enumerate(unique_players, start=1):
        first_row = agent_logs_df[agent_logs_df['player.name'] == player_name].iloc[0]

        player_dict = {
            'name': player_name,
            'model': first_row['player.model'],
            'identity': first_row['player.identity'],
            'tasks': player_tasks.get(player_name, []),
        }

        player_key = f"Player {i}"
        if player_key in players_dict:
            player_dict['narrative'] = create_player_narrative(
                players_dict[player_key],
                player_name,
            )

        players_list.append(player_dict)

    game_id = (
        agent_logs_df['game_index'].iloc[0] if 'game_index' in agent_logs_df.columns else
        agent_logs_df['game_id'].iloc[0] if 'game_id' in agent_logs_df.columns else 0
    )

    return {
        'game_index': game_id,
        'total_steps': int(agent_logs_df['step'].max()),
        'config': {
            'num_players': len(unique_players),
            'num_impostors': sum(1 for p in players_list if p.get('identity') == 'Impostor'),
        },
        'players': players_list,
        'outcome': 'Unknown',
        'winner': 'Unknown',
        'meeting_transcripts': meeting_transcripts,
    }

def get_player_experience_str(game_data, player_id):
    game_config = {
        'game_id': game_data.get('game_index'),
        'total_steps': game_data.get('total_steps'),
        'num_players': len(game_data.get('players', [])),
        'impostor_count': sum(1 for p in game_data.get('players', []) if p.get('identity') == 'Impostor'),
        'outcome': game_data.get('outcome'),
        'winner': game_data.get('winner')
    }

    player_info = None
    for player in game_data.get('players', []):
        if player.get('name') == player_id:
            player_info = {
                'name': player.get('name'),
                'identity': player.get('identity'),
                'tasks': player.get('tasks', [])
            }
            narrative = player.get('narrative', 'Narrative not available')
            break

    if player_info is None:
        return None

    out = f"PLAYER: {player_id}\n\nGAME CONFIGURATION:\n"
    for key, value in game_config.items():
        out += f"  {key}: {value}\n"
    out += "\nPLAYER INFORMATION:\n"
    for key, value in player_info.items():
        out += f"  {key}: {value}\n"
    out += "NARRATIVE:\n"
    out += narrative

    return out



if __name__ == '__main__':
    games = load_all_games()
    first_folder = next(iter(games))
    entries = games[first_folder]
    result = parse_game_logs(entries)
    game_data = create_game_log(entries, result['agent_logs_df'], result['players'], result['meeting_transcripts'])
    experience_string = get_player_experience_str(game_data, "Player 7: cyan")
    print(experience_string)
