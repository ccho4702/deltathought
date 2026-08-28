# Dataset license acceptance records

Do not place credentials here. For a dataset whose official terms require user acceptance, add the
JSON record named in `configs/data.yaml` only after the user has personally accepted those terms.

Required fields:

```json
{
  "dataset": "official dataset name",
  "terms_url": "official terms URL",
  "accepted_at_utc": "ISO-8601 timestamp",
  "accepted_by": "local user identifier without credentials"
}
```

This record documents authorization; it does not replace the upstream license or terms.

For the active long-video pipeline:

- Ego4D requires `ego4d.accepted.json` after personally completing the agreement linked from
  <https://ego4d-data.org/docs/start-here/>.
- LongVideoBench requires `longvideobench.accepted.json` after accepting the gated conditions at
  <https://huggingface.co/datasets/longvideobench/LongVideoBench>.

Set `dataset` to `Ego4D` or `LongVideoBench` exactly. Never copy access tokens or temporary AWS
credentials into either record.
